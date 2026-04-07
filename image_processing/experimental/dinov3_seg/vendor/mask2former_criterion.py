# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def sample_point(
    input_features: torch.Tensor,
    point_coordinates: torch.Tensor,
    add_dim: bool = False,
    **kwargs,
) -> torch.Tensor:
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    point_features = F.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


def dice_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tensor:
    probs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / max(float(num_masks), 1.0)


def sigmoid_cross_entropy_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / max(float(num_masks), 1.0)


def pair_wise_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.matmul(inputs, targets.T)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    height_and_width = inputs.shape[1]
    loss_pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    loss_neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    return torch.matmul(loss_pos / height_and_width, targets.T) + torch.matmul(
        loss_neg / height_and_width, (1 - targets).T
    )


def softmax_focal_loss(
    inputs: Tensor,
    labels: Tensor,
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    normalizer: Optional[float] = None,
) -> Tensor:
    log_probs = F.log_softmax(inputs, dim=1)
    probs = log_probs.exp()

    target_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    loss = -target_log_probs * ((1 - target_probs) ** gamma)

    if alpha is not None:
        loss = loss * alpha[labels]

    if normalizer is not None:
        if isinstance(normalizer, Tensor):
            normalizer = normalizer.detach().float().item()
        return loss.sum() / max(float(normalizer), 1.0)
    return loss.mean()


def weight_reduce_loss(
    loss: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> Tensor:
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError("avg_factor can only be used with reduction='mean'")
    return loss.sum() / max(float(avg_factor), 1.0)


def seesaw_ce_loss(
    cls_score: Tensor,
    labels: Tensor,
    label_weights: Optional[Tensor],
    cum_samples: Tensor,
    num_classes: int,
    p: float,
    q: float,
    eps: float,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> Tensor:
    if cls_score.size(-1) != num_classes:
        raise ValueError(
            f"Seesaw loss expects {num_classes} class logits, but got {cls_score.size(-1)}."
        )
    if cum_samples.numel() != num_classes:
        raise ValueError(
            f"Seesaw cumulative samples must have length {num_classes}, but got {cum_samples.numel()}."
        )

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[torch.arange(len(scores), device=scores.device), labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    adjusted_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))
    loss = F.cross_entropy(adjusted_score, labels, reduction="none")

    if label_weights is not None:
        label_weights = label_weights.float()
    return weight_reduce_loss(
        loss,
        weight=label_weights,
        reduction=reduction,
        avg_factor=avg_factor,
    )


class Mask2FormerHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12_544,
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("At least one Mask2Former matching cost must be non-zero.")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]

        indices: List[Tuple[np.ndarray, np.ndarray]] = []
        for batch_index in range(pred_logits.shape[0]):
            target_labels = targets[batch_index]["labels"]
            target_masks = targets[batch_index]["masks"]
            if target_labels.numel() == 0:
                empty = np.empty(0, dtype=np.int64)
                indices.append((empty, empty))
                continue

            pred_probs = pred_logits[batch_index].softmax(-1)
            pred_mask = pred_masks[batch_index][:, None]
            target_mask = target_masks.to(pred_mask)[:, None]

            cost_class = -pred_probs[:, target_labels]

            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_sample_coords = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            pred_sample_coords = point_coordinates.repeat(pred_mask.shape[0], 1, 1)

            target_mask = sample_point(target_mask, target_sample_coords, align_corners=False).squeeze(1)
            pred_mask = sample_point(pred_mask, pred_sample_coords, align_corners=False).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)

            cost_matrix = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [
            (
                torch.as_tensor(src, dtype=torch.int64, device=pred_logits.device),
                torch.as_tensor(tgt, dtype=torch.int64, device=pred_logits.device),
            )
            for src, tgt in indices
        ]


class Mask2FormerSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        num_points: int = 12_544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        ignore_index: int = 255,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_class_alphas: Optional[Sequence[float]] = None,
        focal_no_object_alpha: Optional[float] = None,
        focal_gamma: float = 2.0,
        focal_normalize_by_num_masks: bool = True,
        classification_loss_type: Optional[str] = None,
        seesaw_p: float = 0.8,
        seesaw_q: float = 2.0,
        seesaw_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.no_object_weight = no_object_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_class_alphas = focal_class_alphas
        self.focal_no_object_alpha = focal_no_object_alpha
        self.focal_gamma = focal_gamma
        self.focal_normalize_by_num_masks = focal_normalize_by_num_masks
        self.classification_loss_type = classification_loss_type
        if self.classification_loss_type is None:
            self.classification_loss_type = "focal" if self.use_focal_loss else "ce"
        self.classification_loss_type = str(self.classification_loss_type).lower()
        if self.classification_loss_type not in {"ce", "focal", "seesaw"}:
            raise ValueError(
                "classification_loss_type must be one of {'ce', 'focal', 'seesaw'}. "
                f"Got {self.classification_loss_type!r}."
            )
        self.seesaw_p = seesaw_p
        self.seesaw_q = seesaw_q
        self.seesaw_eps = seesaw_eps

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
        )

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("focal_alpha_weights", self._build_focal_alpha_weights())
        self.register_buffer("seesaw_cum_samples", torch.zeros(self.num_classes, dtype=torch.float32))

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def forward(self, outputs: Dict[str, Tensor], gt: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        outputs = self._cast_outputs_to_float(outputs)
        targets = self.prepare_targets(gt)
        loss_dict = self.compute_loss_dict(outputs, targets)
        total_loss = outputs["pred_logits"].sum() * 0.0
        for key, value in loss_dict.items():
            if key.startswith("loss_ce"):
                total_loss = total_loss + value * self.class_weight
            elif key.startswith("loss_mask"):
                total_loss = total_loss + value * self.mask_weight
            elif key.startswith("loss_dice"):
                total_loss = total_loss + value * self.dice_weight
        return total_loss, loss_dict

    def _cast_outputs_to_float(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs.float() if outputs.is_floating_point() else outputs
        if isinstance(outputs, dict):
            return {key: self._cast_outputs_to_float(value) for key, value in outputs.items()}
        if isinstance(outputs, list):
            return [self._cast_outputs_to_float(value) for value in outputs]
        if isinstance(outputs, tuple):
            return tuple(self._cast_outputs_to_float(value) for value in outputs)
        return outputs

    def prepare_targets(self, gt: Tensor) -> List[Dict[str, Tensor]]:
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0]
        elif gt.ndim == 2:
            gt = gt.unsqueeze(0)

        targets: List[Dict[str, Tensor]] = []
        for per_image_gt in gt.long():
            class_labels = torch.unique(per_image_gt)
            class_labels = class_labels[class_labels != self.ignore_index]
            if class_labels.numel() == 0:
                mask_labels = per_image_gt.new_zeros(
                    (0, per_image_gt.shape[-2], per_image_gt.shape[-1]),
                    dtype=torch.float32,
                )
            else:
                mask_labels = (per_image_gt.unsqueeze(0) == class_labels[:, None, None]).to(torch.float32)
            targets.append({"labels": class_labels.to(torch.int64), "masks": mask_labels})
        return targets

    def _build_focal_alpha_weights(self) -> Tensor:
        if self.focal_class_alphas is None:
            alpha_weights = torch.full(
                (self.num_classes + 1,),
                float(self.focal_alpha),
                dtype=torch.float32,
            )
            no_object_alpha = self.focal_no_object_alpha
            if no_object_alpha is None:
                no_object_alpha = 1.0 - float(self.focal_alpha)
            alpha_weights[-1] = float(no_object_alpha)
            return alpha_weights

        alpha_weights = torch.as_tensor(self.focal_class_alphas, dtype=torch.float32)
        if alpha_weights.numel() == self.num_classes:
            no_object_alpha = self.focal_no_object_alpha
            if no_object_alpha is None:
                no_object_alpha = 1.0 - float(self.focal_alpha)
            alpha_weights = torch.cat(
                [
                    alpha_weights,
                    torch.tensor([float(no_object_alpha)], dtype=torch.float32),
                ],
                dim=0,
            )
        elif alpha_weights.numel() != self.num_classes + 1:
            raise ValueError(
                "focal_class_alphas must have length num_classes or num_classes + 1. "
                f"Got {alpha_weights.numel()} for num_classes={self.num_classes}."
            )

        if self.focal_no_object_alpha is not None:
            alpha_weights[-1] = float(self.focal_no_object_alpha)
        return alpha_weights

    def compute_loss_dict(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = self.get_num_masks(targets, device=outputs["pred_logits"].device)

        losses = {}
        losses.update(self.loss_labels(outputs_without_aux, targets, indices, num_masks))
        losses.update(self.loss_masks(outputs_without_aux, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_losses = {}
                aux_losses.update(
                    self.loss_labels(
                        aux_outputs,
                        targets,
                        aux_indices,
                        num_masks,
                        update_seesaw_statistics=False,
                    )
                )
                aux_losses.update(self.loss_masks(aux_outputs, targets, aux_indices, num_masks))
                losses.update({f"{k}_{idx}": v for k, v in aux_losses.items()})

        return losses

    def loss_labels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_masks: float,
        update_seesaw_statistics: bool = True,
    ) -> Dict[str, Tensor]:
        src_logits = outputs["pred_logits"].float()
        batch_size, num_queries, _ = src_logits.shape

        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        batch_idx, src_idx = self._get_src_permutation_idx(indices, device=src_logits.device)
        matched_classes = [target["labels"][j] for target, (_, j) in zip(targets, indices) if j.numel() > 0]
        if matched_classes:
            target_classes_o = torch.cat(matched_classes)
            target_classes[batch_idx, src_idx] = target_classes_o

        pred_logits_transposed = src_logits.transpose(1, 2)
        if self.classification_loss_type == "seesaw":
            loss_ce = self._seesaw_classification_loss(
                src_logits,
                target_classes,
                update_statistics=update_seesaw_statistics,
            )
        elif self.classification_loss_type == "focal":
            normalizer = num_masks if self.focal_normalize_by_num_masks else None
            loss_ce = softmax_focal_loss(
                pred_logits_transposed,
                target_classes,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha_weights.to(src_logits.device),
                normalizer=normalizer,
            )
        else:
            loss_ce = F.cross_entropy(pred_logits_transposed, target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_masks: float,
    ) -> Dict[str, Tensor]:
        src_masks = outputs["pred_masks"]
        batch_idx, src_idx = self._get_src_permutation_idx(indices, device=src_masks.device)
        tgt_batch_idx, tgt_idx = self._get_tgt_permutation_idx(indices, device=src_masks.device)

        if src_idx.numel() == 0:
            zero = src_masks.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        pred_masks = src_masks[batch_idx, src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch([t["masks"] for t in targets])
        target_masks = target_masks.to(pred_masks)
        target_masks = target_masks[tgt_batch_idx, tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                self.calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)
        return {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

    def get_num_masks(self, targets: List[Dict[str, Tensor]], device: torch.device) -> float:
        num_masks = torch.as_tensor([sum(len(target["labels"]) for target in targets)], dtype=torch.float, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks)
            num_masks = num_masks / dist.get_world_size()
        return max(num_masks.item(), 1.0)

    def _update_seesaw_cum_samples(self, labels: Tensor) -> None:
        if (not self.training) or labels.numel() == 0:
            return

        with torch.no_grad():
            batch_counts = torch.bincount(labels.detach().reshape(-1), minlength=self.num_classes).to(
                device=self.seesaw_cum_samples.device,
                dtype=self.seesaw_cum_samples.dtype,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_counts)
            self.seesaw_cum_samples.add_(batch_counts)

    def _seesaw_classification_loss(
        self,
        pred_logits: Tensor,
        target_classes: Tensor,
        update_statistics: bool = True,
    ) -> Tensor:
        flat_logits = pred_logits.reshape(-1, pred_logits.shape[-1]).float()
        flat_targets = target_classes.reshape(-1)
        foreground_mask = flat_targets != self.num_classes
        no_object_mask = ~foreground_mask

        total_loss = flat_logits.new_zeros(())
        total_weight = flat_logits.new_zeros(())

        if foreground_mask.any():
            foreground_logits = flat_logits[foreground_mask, : self.num_classes]
            foreground_targets = flat_targets[foreground_mask]
            if update_statistics and self.training:
                self._update_seesaw_cum_samples(foreground_targets)
            foreground_weights = foreground_logits.new_ones(
                foreground_targets.shape[0],
                dtype=torch.float32,
            )
            foreground_loss = seesaw_ce_loss(
                foreground_logits,
                foreground_targets,
                foreground_weights,
                self.seesaw_cum_samples,
                self.num_classes,
                self.seesaw_p,
                self.seesaw_q,
                self.seesaw_eps,
                reduction="sum",
            )
            total_loss = total_loss + foreground_loss
            total_weight = total_weight + foreground_weights.sum()

        if no_object_mask.any():
            no_object_logits = flat_logits[no_object_mask]
            no_object_targets = flat_targets[no_object_mask]
            no_object_loss = F.cross_entropy(no_object_logits, no_object_targets, reduction="sum")
            total_loss = total_loss + self.no_object_weight * no_object_loss
            total_weight = total_weight + flat_logits.new_tensor(
                float(no_object_targets.numel()) * float(self.no_object_weight)
            )

        return total_loss / total_weight.clamp(min=1.0)

    def calculate_uncertainty(self, logits: Tensor) -> Tensor:
        return -torch.abs(logits)

    def sample_points_using_uncertainty(
        self,
        logits: Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> Tensor:
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx = idx + shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        max_size = [max(sizes) for sizes in zip(*[list(tensor.shape) for tensor in tensors])]
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False
        return padded_tensors, padding_masks

    def _get_src_permutation_idx(
        self,
        indices: List[Tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        src_indices = [src for src, _ in indices if src.numel() > 0]
        if not src_indices:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        batch_indices = [torch.full_like(src, batch_index) for batch_index, (src, _) in enumerate(indices) if src.numel() > 0]
        return torch.cat(batch_indices), torch.cat(src_indices)

    def _get_tgt_permutation_idx(
        self,
        indices: List[Tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        tgt_indices = [tgt for _, tgt in indices if tgt.numel() > 0]
        if not tgt_indices:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        batch_indices = [torch.full_like(tgt, batch_index) for batch_index, (_, tgt) in enumerate(indices) if tgt.numel() > 0]
        return torch.cat(batch_indices), torch.cat(tgt_indices)

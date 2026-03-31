from typing import Tuple

import torch
import torch.nn.functional as F


def outputs_to_semantic_predictions(
        outputs, target_size: Tuple[int, int]) -> torch.Tensor:
    class_logits = outputs.class_queries_logits[..., :-1]
    mask_logits = outputs.masks_queries_logits

    class_probs = torch.softmax(class_logits, dim=-1)
    mask_probs = torch.sigmoid(mask_logits)
    semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    if semantic_logits.shape[-2:] != target_size:
        semantic_logits = F.interpolate(
            semantic_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return semantic_logits.argmax(dim=1)


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return

    filtered_targets = targets[valid_mask].to(torch.int64)
    filtered_predictions = predictions[valid_mask].to(torch.int64)

    encoded = filtered_targets * num_classes + filtered_predictions
    bins = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion_matrix += bins.reshape(num_classes,
                                     num_classes).to(confusion_matrix.device)


def compute_mean_iou(confusion_matrix: torch.Tensor) -> float:
    true_positive = torch.diag(confusion_matrix)
    false_positive = confusion_matrix.sum(dim=0) - true_positive
    false_negative = confusion_matrix.sum(dim=1) - true_positive
    union = true_positive + false_positive + false_negative

    valid = union > 0
    if not torch.any(valid):
        return 0.0

    iou = true_positive[valid].float() / union[valid].float()
    return float(iou.mean().item())

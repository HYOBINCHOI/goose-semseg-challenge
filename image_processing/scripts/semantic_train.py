"""
Training code for a tuned ConvNeXt + Mask2Former model.
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEFAULT_GOOSE_TOOLS_ROOT = str(
    (Path(__file__).resolve().parent.parent / "image_processing").resolve()
)


def seed_everything(seed: int) -> None: # Seed all RNGs so data shuffling and training are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_goose_dataset_class(goose_tools_root: str):
    goose_root = Path(goose_tools_root).resolve()
    if not goose_root.exists():
        raise FileNotFoundError(f"goose_tools_root does not exist: {goose_root}")

    if str(goose_root) not in sys.path:
        sys.path.insert(0, str(goose_root))

    from goosetools import GOOSE_Dataset

    return GOOSE_Dataset


def resolve_goose_data_root(data_path: str) -> Path:
    requested_root = Path(data_path).expanduser().resolve()
    candidate_roots = [requested_root, requested_root / "goose-dataset"]

    for root in candidate_roots:
        if (root / "images" / "train").is_dir() and (root / "labels" / "train").is_dir():
            return root

    checked_roots = ", ".join(str(root) for root in candidate_roots)
    raise FileNotFoundError(
        "Could not find a valid GOOSE dataset root. "
        "Expected 'images/train' and 'labels/train' under one of: "
        f"{checked_roots}"
    )


def default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        "/home/mipstu/jiPark/challenge/goose_dataset/output/"
        f"{timestamp}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ConvNeXt + Mask2Former Trainer (boosted)")

    parser.add_argument("data_path", type=str, help="Path to goose dataset root")
    parser.add_argument(
        "--goose_tools_root",
        type=str,
        default=DEFAULT_GOOSE_TOOLS_ROOT,
        help="Directory that contains the goosetools package.",
    )
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument(
        "--run_name", type=str, default="convnext_mask2former"
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--resize_width", type=int, default=1024)
    parser.add_argument("--resize_height", type=int, default=1024)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument(
        "--convnext_model_name_or_path",
        type=str,
        default="facebook/dinov3-convnext-large-pretrain-lvd1689m",
        help="Pretrained ConvNeXt backbone checkpoint.",
    )
    parser.add_argument(
        "--mask2former_pretrained_model_name_or_path",
        type=str,
        default="facebook/mask2former-swin-large-ade-semantic",
        help="Pretrained Mask2Former checkpoint for decoder/heads initialization.",
    )
    parser.add_argument(
        "--freeze_encoder",
        dest="freeze_encoder",
        action="store_true",
        help="Freeze the ConvNeXt encoder.",
    )
    parser.add_argument(
        "--unfreeze_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Train the full ConvNeXt encoder with a lower lr.",
    )
    parser.add_argument(
        "--freeze_mask2former_decoder",
        action="store_true",
        help="Freeze pretrained Mask2Former transformer decoder and prediction heads.",
    )

    parser.add_argument(
        "--feature_indices",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="ConvNeXt stage indices selected as multi-scale features.",
    )
    parser.add_argument(
        "--adapter_hidden_dim",
        type=int,
        default=256,
        help="Intermediate channel size in the ConvNeXt-to-Mask2Former adapter.",
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.10,
        help="Spatial dropout used inside each adapter block.",
    )
    parser.add_argument(
        "--fusion_dropout",
        type=float,
        default=0.10,
        help="Spatial dropout used in the mask-feature fusion head.",
    )
    parser.add_argument(
        "--mask_feature_fusion_levels",
        type=int,
        default=2,
        help="Number of highest-resolution levels fused for mask feature generation.",
    )

    parser.add_argument(
        "--disable_encoder_norm",
        action="store_true",
        help="Disable ConvNeXt normalization from AutoImageProcessor stats.",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint created by this script.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=1)

    parser.set_defaults(freeze_encoder=True)
    return parser.parse_args()


def semantic_map_to_targets(
    semantic_map: torch.Tensor, ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(semantic_map)
    classes = classes[classes != ignore_index]

    if classes.numel() == 0:
        empty_classes = torch.zeros((0,), dtype=torch.long)
        empty_masks = torch.zeros(
            (0, semantic_map.shape[0], semantic_map.shape[1]),
            dtype=torch.float32,
        )
        return empty_classes, empty_masks

    masks = [(semantic_map == class_idx).to(torch.float32) for class_idx in classes]
    return classes.to(torch.long), torch.stack(masks, dim=0)


def outputs_to_semantic_predictions(
    outputs, target_size: Tuple[int, int]
) -> torch.Tensor:
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
    confusion_matrix += bins.reshape(num_classes, num_classes).to(
        confusion_matrix.device
    )


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


def save_checkpoint(
    save_path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_val_loss: float,
    best_val_miou: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_miou": best_val_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, save_path)


def metric_checkpoint_path(
    run_dir: Path, prefix: str, epoch: int, score: float, higher_is_better: bool
) -> Path:
    metric_name = "miou" if higher_is_better else "loss"
    return run_dir / f"{prefix}_epoch_{epoch + 1:03d}_{metric_name}_{score:.4f}.pt"


def ensure_epoch_log_file(log_path: Path) -> None:
    if log_path.exists():
        return

    with open(log_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_miou",
                "val_loss",
                "val_miou",
                "best_val_loss",
                "best_val_miou",
            ]
        )


def append_epoch_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    train_miou: float,
    val_loss: float,
    val_miou: float,
    best_val_loss: float,
    best_val_miou: float,
) -> None:
    with open(log_path, "a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                epoch + 1,
                f"{train_loss:.6f}",
                f"{train_miou:.6f}",
                f"{val_loss:.6f}",
                f"{val_miou:.6f}",
                f"{best_val_loss:.6f}",
                f"{best_val_miou:.6f}",
            ]
        )


def save_training_curves(log_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping training curve export.")
        return

    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_mious: List[float] = []
    val_mious: List[float] = []

    with open(log_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_mious.append(float(row["train_miou"]))
            val_mious.append(float(row["val_miou"]))

    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="train_loss", marker="o")
    axes[0].plot(epochs, val_losses, label="val_loss", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_mious, label="train_mIoU", marker="o")
    axes[1].plot(epochs, val_mious, label="val_mIoU", marker="o")
    axes[1].set_title("mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class GooseMask2FormerCollator:
    def __init__(self, ignore_index: int):
        self.ignore_index = ignore_index

    def __call__(
        self, batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, object]:
        images, semantic_maps = zip(*batch)
        class_labels: List[torch.Tensor] = []
        mask_labels: List[torch.Tensor] = []

        for semantic_map in semantic_maps:
            classes, masks = semantic_map_to_targets(semantic_map, self.ignore_index)
            class_labels.append(classes)
            mask_labels.append(masks)

        return {
            "pixel_values": torch.stack(images),
            "semantic_maps": torch.stack(semantic_maps),
            "class_labels": class_labels,
            "mask_labels": mask_labels,
        }


def move_batch_to_device(
    batch: Dict[str, object], device: torch.device
) -> Dict[str, object]:
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "semantic_maps": batch["semantic_maps"].to(device, non_blocking=True),
        "class_labels": [item.to(device) for item in batch["class_labels"]],
        "mask_labels": [item.to(device) for item in batch["mask_labels"]],
    }


class DINOInputNormalizer(nn.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float], enabled: bool):
        super().__init__()
        self.enabled = enabled
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.max().detach().item() > 1.5:
            x = x / 255.0
        return (x - self.mean) / self.std


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> AdamW:
    encoder_parameters: List[nn.Parameter] = []
    other_parameters: List[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "pixel_level_module.encoder" in name:
            encoder_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []
    if other_parameters:
        parameter_groups.append(
            {
                "params": other_parameters,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": args.encoder_lr,
                "weight_decay": args.weight_decay,
            }
        )

    return AdamW(parameter_groups)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: Optional[AdamW],
    use_amp: bool,
    grad_clip_norm: float,
    num_classes: int,
    ignore_index: int,
    epoch_index: int,
    total_epochs: int,
    global_progress: tqdm,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch_index + 1}/{total_epochs} {'train' if is_train else 'val'}",
        dynamic_ncols=True,
        leave=False,
        position=1,
    )

    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    class_labels=batch["class_labels"],
                    mask_labels=batch["mask_labels"],
                )
                loss = outputs.loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())

        predictions = outputs_to_semantic_predictions(
            outputs, target_size=batch["semantic_maps"].shape[-2:]
        )
        update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            predictions=predictions,
            targets=batch["semantic_maps"],
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        progress_bar.set_postfix(
            loss=f"{total_loss / step:.4f}",
            miou=f"{compute_mean_iou(confusion_matrix):.4f}",
        )

        global_progress.update(1)
        global_progress.set_postfix(
            epoch=f"{epoch_index + 1}/{total_epochs}",
            phase="train" if is_train else "val",
            step=f"{step}/{len(loader)}",
            refresh=False,
        )

    return total_loss / max(len(loader), 1), compute_mean_iou(confusion_matrix)


def resume_if_needed(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float, float]:
    if not args.resume_from:
        return 0, float("inf"), float("-inf")

    checkpoint = torch.load(args.resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_miou = float(checkpoint.get("best_val_miou", float("-inf")))
    print(f"Resumed from epoch {start_epoch}.")
    return start_epoch, best_val_loss, best_val_miou


class AdapterProjectionBlock(nn.Module): #Projects each ConvNeXt feature map into the feature space expected by Mask2Former
    def __init__( 
        self, 
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


class MaskFeatureFusionHead(nn.Module): #Fuses multi-scale features into a high-resolution mask feature representation
    def __init__(
        self,
        feature_dim: int,
        adapter_hidden_dim: int,
        mask_feature_dim: int,
        fusion_levels: int,
        dropout: float,
    ):
        super().__init__()
        self.fusion_levels = fusion_levels
        in_channels = feature_dim * fusion_levels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, adapter_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(adapter_hidden_dim, adapter_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(adapter_hidden_dim, mask_feature_dim, kernel_size=1),
        )

    def forward(self, multi_scale_features: Sequence[torch.Tensor]) -> torch.Tensor:
        levels = list(multi_scale_features[: self.fusion_levels])
        if not levels:
            raise ValueError("Expected at least one feature level for mask fusion.")

        base_size = levels[0].shape[-2:]
        resized_levels = [levels[0]]
        for feature in levels[1:]:
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(
                    feature,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_levels.append(feature)
        return self.fusion(torch.cat(resized_levels, dim=1))


class ConvNeXtPixelLevelModuleBoosted(nn.Module): #Replaces Mask2Former's pixel-level module with a ConvNeXt-based multi-scale feature extractor
    def __init__(
        self,
        encoder: nn.Module,
        normalizer: nn.Module,
        feature_indices: Sequence[int],
        encoder_hidden_sizes: Sequence[int],
        feature_dim: int,
        mask_feature_dim: int,
        adapter_hidden_dim: int,
        adapter_dropout: float,
        fusion_dropout: float,
        mask_feature_fusion_levels: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.normalizer = normalizer
        self.feature_indices = list(feature_indices)
        self.encoder_hidden_sizes = list(encoder_hidden_sizes)

        self.input_projections = nn.ModuleList(
            [
                AdapterProjectionBlock(
                    in_channels=stage_hidden_size,
                    hidden_channels=adapter_hidden_dim,
                    out_channels=feature_dim,
                    dropout=adapter_dropout,
                )
                for stage_hidden_size in encoder_hidden_sizes
            ]
        )
        self.level_scales = nn.Parameter(torch.ones(len(encoder_hidden_sizes)))
        self.mask_fusion = MaskFeatureFusionHead(
            feature_dim=feature_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            mask_feature_dim=mask_feature_dim,
            fusion_levels=min(mask_feature_fusion_levels, len(encoder_hidden_sizes)),
            dropout=fusion_dropout,
        )

    def _select_feature_maps( 
        self, feature_maps: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:

        if len(feature_maps) == len(self.feature_indices):
            for feature_map in feature_maps:
                if feature_map.ndim != 4:
                    raise ValueError(
                        "Expected each ConvNeXt feature map to be 4D, "
                        f"but got shape {tuple(feature_map.shape)}."
                    )
            return list(feature_maps)

        selected = []
        for index in self.feature_indices:
            if index >= len(feature_maps) or index < -len(feature_maps):
                raise IndexError(
                    f"feature index {index} is out of range for "
                    f"{len(feature_maps)} ConvNeXt feature maps"
                )
            feature_map = feature_maps[index]
            if feature_map.ndim != 4:
                raise ValueError(
                    f"Expected ConvNeXt feature map at index {index} to be 4D, "
                    f"but got shape {tuple(feature_map.shape)}."
                )
            selected.append(feature_map)
        return selected

    def _extract_feature_maps(self, encoder_outputs) -> Sequence[torch.Tensor]: 
        feature_maps = getattr(encoder_outputs, "feature_maps", None)
        if feature_maps is not None:
            return feature_maps

        hidden_states = getattr(encoder_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "The selected ConvNeXt checkpoint did not return feature maps or hidden states."
            )

        valid_stage_channels = set(self.encoder_hidden_sizes)
        spatial_hidden_states = [
            hidden_state
            for hidden_state in hidden_states
            if hidden_state.ndim == 4 and hidden_state.shape[1] in valid_stage_channels
        ]
        if not spatial_hidden_states:
            raise ValueError(
                "The selected ConvNeXt checkpoint returned hidden states, but none matched the expected ConvNeXt stage channels."
            )

        deduplicated_feature_maps: List[torch.Tensor] = []
        for hidden_state in spatial_hidden_states:
            if (
                deduplicated_feature_maps
                and deduplicated_feature_maps[-1].shape[-2:] == hidden_state.shape[-2:]
            ):
                deduplicated_feature_maps[-1] = hidden_state
            else:
                deduplicated_feature_maps.append(hidden_state)

        deduplicated_feature_maps = [
            hidden_state
            for hidden_state in deduplicated_feature_maps
            if hidden_state.shape[1] in valid_stage_channels
        ]
        return tuple(deduplicated_feature_maps)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        try:
            from transformers.models.mask2former.modeling_mask2former import (
                Mask2FormerPixelLevelModuleOutput,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with `pip install transformers`."
            ) from exc

        normalized_pixel_values = self.normalizer(pixel_values)

        encoder_outputs = self.encoder(
            pixel_values=normalized_pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_maps = self._extract_feature_maps(encoder_outputs)
        selected_feature_maps = self._select_feature_maps(feature_maps)

        multi_scale_features: List[torch.Tensor] = []
        base_height, base_width = selected_feature_maps[0].shape[-2:]

        for level, (projection, feature_map) in enumerate(
            zip(self.input_projections, selected_feature_maps)
        ):
            target_height = max(base_height // (2**level), 1)
            target_width = max(base_width // (2**level), 1)

            if feature_map.shape[-2:] != (target_height, target_width):
                feature_map = F.interpolate(
                    feature_map,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )

            projected = projection(feature_map) * self.level_scales[level]
            multi_scale_features.append(projected)

        mask_features = self.mask_fusion(multi_scale_features)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=multi_scale_features[-1],
            encoder_hidden_states=tuple(multi_scale_features),
            decoder_last_hidden_state=mask_features,
            decoder_hidden_states=tuple(multi_scale_features),
        )


class ConvNeXtMask2FormerBoostedModel(nn.Module): #Load a pretrained ConvNeXt and a pretrained Mask2Former, then combine them into a single model
    def __init__(
        self,
        args: argparse.Namespace,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ):
        super().__init__()
        try:
            from transformers import (
                AutoConfig,
                AutoBackbone,
                AutoImageProcessor,
                AutoModel,
                Mask2FormerForUniversalSegmentation,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with `pip install transformers>=4.56.0`."
            ) from exc

        encoder_config = AutoConfig.from_pretrained(
            args.convnext_model_name_or_path,
            local_files_only=True,
        )
        out_indices = tuple(sorted(set(args.feature_indices)))

        if encoder_config.model_type == "dinov3_convnext":
            encoder = AutoModel.from_pretrained(
                args.convnext_model_name_or_path,
                local_files_only=True,
            )
        else:
            encoder = AutoBackbone.from_pretrained(
                args.convnext_model_name_or_path,
                out_indices=out_indices,
                local_files_only=True,
            )

        if args.freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        try:
            processor = AutoImageProcessor.from_pretrained(
                args.convnext_model_name_or_path,
                local_files_only=True,
            )
            image_mean = getattr(processor, "image_mean", image_mean)
            image_std = getattr(processor, "image_std", image_std)
        except Exception as exc:
            print(
                "Warning: failed to load ConvNeXt image processor from "
                f"{args.convnext_model_name_or_path}. "
                "Falling back to default ImageNet normalization stats. "
                f"Original error: {exc}"
            )

        normalizer = DINOInputNormalizer(
            mean=image_mean,
            std=image_std,
            enabled=not args.disable_encoder_norm,
        )

        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            args.mask2former_pretrained_model_name_or_path,
            num_labels=args.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

        feature_dim = self.mask2former.config.feature_size
        mask_feature_dim = self.mask2former.config.mask_feature_size

        hidden_sizes = list(getattr(encoder.config, "hidden_sizes", []))
        if not hidden_sizes:
            raise ValueError(
                "Could not infer ConvNeXt hidden_sizes from the encoder config."
            )

        encoder_hidden_sizes = []
        for index in args.feature_indices:
            if index >= len(hidden_sizes) or index < -len(hidden_sizes):
                raise IndexError(
                    f"feature index {index} is out of range for ConvNeXt hidden sizes "
                    f"{hidden_sizes}"
                )
            encoder_hidden_sizes.append(hidden_sizes[index])

        self.mask2former.model.pixel_level_module = ConvNeXtPixelLevelModuleBoosted(  # Replaces the original pixel-level module with the ConvNeXt-based module
            encoder=encoder,
            normalizer=normalizer,
            feature_indices=args.feature_indices,
            encoder_hidden_sizes=encoder_hidden_sizes,
            feature_dim=feature_dim,
            mask_feature_dim=mask_feature_dim,
            adapter_hidden_dim=args.adapter_hidden_dim,
            adapter_dropout=args.adapter_dropout,
            fusion_dropout=args.fusion_dropout,
            mask_feature_fusion_levels=args.mask_feature_fusion_levels,
        )

        if args.freeze_mask2former_decoder:
            for name, parameter in self.mask2former.named_parameters():
                if not name.startswith("model.pixel_level_module."):
                    parameter.requires_grad = False

    def forward(self, **kwargs):
        return self.mask2former(**kwargs)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    goose_dataset_class = load_goose_dataset_class(args.goose_tools_root)
    data_root = resolve_goose_data_root(args.data_path)

    train_dataset, val_dataset = goose_dataset_class.splits_from_path(
        str(data_root),
        resize_size=[args.resize_width, args.resize_height],
        crop=args.crop,
    )

    print(f"Resolved GOOSE dataset root: {data_root}")
    print(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
    )

    collator = GooseMask2FormerCollator(ignore_index=args.ignore_index) 
    pin_memory = device.type == "cuda" and not args.disable_pin_memory
    persistent_workers = args.persistent_workers and args.num_workers > 0

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True, 
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "persistent_workers": persistent_workers,
    }
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    id2label = {i: f"class_{i}" for i in range(args.num_classes)} 
    label2id = {label: idx for idx, label in id2label.items()}

    model = ConvNeXtMask2FormerBoostedModel(args, id2label, label2id).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )

    run_dir = Path(args.output_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_path = run_dir / "epoch_metrics.csv"
    curve_path = run_dir / "training_curves.png"

    with open(run_dir / "train_args.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)
    ensure_epoch_log_file(epoch_log_path)

    start_epoch, best_val_loss, best_val_miou = resume_if_needed(
        args, model, optimizer, scaler, device
    )

    total_training_steps = (args.epochs - start_epoch) * (
        len(train_loader) + len(val_loader)
    )
    global_progress = tqdm(
        total=total_training_steps,
        desc="Total Progress",
        dynamic_ncols=True,
        position=0,
    )
    epochs_without_improvement = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_miou = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                scaler=scaler,
                optimizer=optimizer,
                use_amp=args.amp,
                grad_clip_norm=args.grad_clip_norm,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                epoch_index=epoch,
                total_epochs=args.epochs,
                global_progress=global_progress,
            )
            val_loss, val_miou = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                scaler=scaler,
                optimizer=None,
                use_amp=args.amp,
                grad_clip_norm=args.grad_clip_norm,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                epoch_index=epoch,
                total_epochs=args.epochs,
                global_progress=global_progress,
            )

            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
                f"val_loss={val_loss:.4f} val_miou={val_miou:.4f}"
            )

            miou_improved = val_miou > (best_val_miou + args.early_stopping_min_delta)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    run_dir / "best_loss.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_miou,
                    args,
                )
                save_checkpoint(
                    metric_checkpoint_path(
                        run_dir=run_dir,
                        prefix="best_loss",
                        epoch=epoch,
                        score=best_val_loss,
                        higher_is_better=False,
                    ),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_miou,
                    args,
                )
                print(f"Saved best loss checkpoint: val_loss={best_val_loss:.4f}")

            if miou_improved:
                best_val_miou = val_miou
                epochs_without_improvement = 0
                save_checkpoint(
                    run_dir / "best_miou.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_miou,
                    args,
                )
                save_checkpoint(
                    metric_checkpoint_path(
                        run_dir=run_dir,
                        prefix="best_miou",
                        epoch=epoch,
                        score=best_val_miou,
                        higher_is_better=True,
                    ),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_miou,
                    args,
                )
                print(f"Saved best mIoU checkpoint: val_miou={best_val_miou:.4f}")
            else:
                epochs_without_improvement += 1

            append_epoch_log(
                log_path=epoch_log_path,
                epoch=epoch,
                train_loss=train_loss,
                train_miou=train_miou,
                val_loss=val_loss,
                val_miou=val_miou,
                best_val_loss=best_val_loss,
                best_val_miou=best_val_miou,
            )
            save_training_curves(epoch_log_path, curve_path)

            save_checkpoint(
                run_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val_loss,
                best_val_miou,
                args,
            )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch + 1}.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_miou,
                    args,
                )

            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epochs without val mIoU improvement."
                )
                break
    finally:
        global_progress.close()


if __name__ == "__main__":
    main()

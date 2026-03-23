"""
GT vs Prediction visualization for ConvNeXt + Mask2Former checkpoints.

This script keeps the original DINO comparison script untouched and provides
the same workflow for ConvNeXt-based checkpoints, especially the boosted
variant used in this project.
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if not hasattr(torch.amp, "GradScaler"):
    torch.amp.GradScaler = torch.cuda.amp.GradScaler  # type: ignore[attr-defined]


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_train_modules():
    m2f_module = load_module("mask2former_train", SCRIPT_DIR / "mask2former_train.py")
    convnext_module = load_module(
        "convnext_mask2former_train_512_64",
        SCRIPT_DIR / "convnext_mask2former_train_512_64.py",
    )
    boosted_module = load_module(
        "convnext_mask2former_train_512_64_boosted",
        SCRIPT_DIR / "convnext_mask2former_train_512_64_boosted.py",
    )
    return m2f_module, convnext_module, boosted_module


M2F_MODULE, CONVNEXT_MODULE, BOOSTED_MODULE = load_train_modules()

ConvNeXtMask2FormerModel = CONVNEXT_MODULE.ConvNeXtMask2FormerModel
ConvNeXtMask2FormerBoostedModel = BOOSTED_MODULE.ConvNeXtMask2FormerBoostedModel
outputs_to_semantic_predictions = M2F_MODULE.outputs_to_semantic_predictions
load_goose_dataset_class = M2F_MODULE.load_goose_dataset_class
DEFAULT_GOOSE_TOOLS_ROOT = M2F_MODULE.DEFAULT_GOOSE_TOOLS_ROOT


def load_colormap(colormap_path: str) -> Dict[int, Tuple[int, int, int]]:
    with open(colormap_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    cmap = {}
    for key, value in raw.items():
        rgb = [int(channel) if channel > 1 else int(channel * 255) for channel in value]
        cmap[int(key)] = tuple(rgb)
    return cmap


def mask_to_color_image(mask: np.ndarray, colormap: Dict[int, Tuple]) -> np.ndarray:
    h, w = mask.shape
    color_img = np.full((h, w, 3), 128, dtype=np.uint8)
    for cls_id, color in colormap.items():
        region = mask == cls_id
        if region.any():
            color_img[region] = color
    return color_img


def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: Dict[int, Tuple],
    alpha: float = 0.5,
) -> np.ndarray:
    color_mask = mask_to_color_image(mask, colormap).astype(np.float32)
    image_float = image.astype(np.float32)
    overlay = image_float * (1 - alpha) + color_mask * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_convnext_model(
    args: argparse.Namespace,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> torch.nn.Module:
    run_name = getattr(args, "run_name", "")
    output_dir = getattr(args, "output_dir", "")
    checkpoint_hint = " ".join([run_name, output_dir]).lower()
    if "boosted" in checkpoint_hint:
        return ConvNeXtMask2FormerBoostedModel(args, id2label, label2id)
    return ConvNeXtMask2FormerModel(args, id2label, label2id)


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[torch.nn.Module, argparse.Namespace]:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    raw_args = ckpt["args"]
    args = argparse.Namespace(**raw_args)

    id2label = {i: f"class_{i}" for i in range(args.num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = build_convnext_model(args, id2label, label2id).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    best_miou = ckpt.get("best_val_miou", "?")
    print(f"  -> Epoch: {epoch}  |  Best val mIoU: {best_miou}")
    return model, args


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int],
) -> np.ndarray:
    pixel_values = image_tensor.unsqueeze(0).to(device)
    outputs = model(pixel_values=pixel_values)
    pred = outputs_to_semantic_predictions(outputs, target_size=target_size)
    return pred.squeeze(0).cpu().numpy().astype(np.int32)


def visualize_comparison(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    colormap: Dict[int, Tuple],
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    image_np = np.array(image.convert("RGB"))

    gt_color = mask_to_color_image(gt_mask, colormap)
    pred_color = mask_to_color_image(pred_mask, colormap)
    gt_overlay = overlay_segmentation(image_np, gt_mask, colormap, alpha=0.5)
    pred_overlay = overlay_segmentation(image_np, pred_mask, colormap, alpha=0.5)

    diff_mask = (gt_mask != pred_mask).astype(np.uint8)
    diff_vis = image_np.copy()
    diff_vis[diff_mask == 1] = [255, 0, 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=12)

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("GT Overlay")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title("Prediction Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gt_color)
    axes[1, 0].set_title("GT Segmentation")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_color)
    axes[1, 1].set_title("Prediction Segmentation")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(diff_vis)
    error_pct = diff_mask.mean() * 100
    axes[1, 2].set_title(f"Error Map (red = wrong, {error_pct:.1f}% pixels wrong)")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def compute_per_class_iou(
    confusion_matrix: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, float]:
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    union = tp + fp + fn
    valid = union > 0
    iou = np.zeros(num_classes)
    iou[valid] = tp[valid] / union[valid]
    miou = iou[valid].mean() if valid.any() else 0.0
    return iou, float(miou)


def parse_args() -> argparse.Namespace:
    default_ckpt = str(
        PROJECT_ROOT
        / "output/convnext_mask2former_512_64_boosted/20260321_184712"
        / "goose_convnext_mask2former_512_64_boosted/best_miou.pt"
    )
    default_colormap = str(PROJECT_ROOT / "common/goose_colormap.json")
    default_output = str(PROJECT_ROOT / "output/comparison_results_convnext")

    parser = argparse.ArgumentParser("GT vs Prediction comparison for ConvNeXt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to a ConvNeXt-based .pt checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/datasets/goose-dataset",
        help="GOOSE dataset root path.",
    )
    parser.add_argument(
        "--goose_tools_root",
        type=str,
        default=DEFAULT_GOOSE_TOOLS_ROOT,
        help="Directory that contains the goosetools package.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default=default_colormap,
        help="Path to goose_colormap.json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to visualize or evaluate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to visualize. Use 0 to skip images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output,
        help="Directory for visualization outputs and CSV files.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval_all",
        action="store_true",
        help="Evaluate the full split and save per-class IoU results.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific dataset indices to visualize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = load_model_from_checkpoint(args.checkpoint, device)

    colormap = load_colormap(args.colormap)
    print(f"Colormap loaded: {len(colormap)} classes")

    print(f"Loading dataset from: {args.data_path} (split={args.split})")
    goose_dataset_class = load_goose_dataset_class(args.goose_tools_root)
    dataset = goose_dataset_class.from_paths(
        img_path=f"{args.data_path}/images/{args.split}",
        lbl_path=f"{args.data_path}/labels/{args.split}",
        resize_size=[model_args.resize_width, model_args.resize_height],
        crop=getattr(model_args, "crop", False),
    )
    print(f"Dataset size: {len(dataset)}")

    if args.indices is not None:
        sample_indices = args.indices
    elif args.num_samples > 0:
        sample_indices = random.sample(
            range(len(dataset)), min(args.num_samples, len(dataset))
        )
    else:
        sample_indices = []

    if sample_indices:
        print(f"\nVisualizing {len(sample_indices)} samples...")
        for idx in tqdm(sample_indices, desc="Generating comparison images"):
            image_tensor, gt_tensor = dataset[idx]
            image_pil, _gt_pil, _instance_pil, _color_pil = dataset.get_images(idx)

            gt_mask = np.array(gt_tensor)
            pred_mask = predict(model, image_tensor, device, gt_mask.shape)

            image_resized = image_pil.resize(
                (model_args.resize_width, model_args.resize_height),
                resample=Image.BILINEAR,
            )

            image_path = dataset.dataset_dict[idx]["img_path"]
            image_name = Path(image_path).stem
            save_path = str(output_dir / f"comparison_{idx:05d}_{image_name}.png")

            visualize_comparison(
                image=image_resized,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                colormap=colormap,
                title=f"[{idx}] {image_name}",
                save_path=save_path,
            )

        print(f"\nSaved {len(sample_indices)} comparison images to: {output_dir}")

    if args.eval_all or not sample_indices:
        print("\nRunning full evaluation on the dataset...")
        num_classes = model_args.num_classes
        ignore_index = model_args.ignore_index
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            image_tensor, gt_tensor = dataset[idx]
            gt_mask = np.array(gt_tensor)
            pred_mask = predict(model, image_tensor, device, gt_mask.shape)

            valid = gt_mask != ignore_index
            gt_valid = gt_mask[valid]
            pred_valid = pred_mask[valid]

            encoded = (gt_valid * num_classes + pred_valid).astype(np.int64)
            encoded = np.clip(encoded, 0, num_classes * num_classes - 1)
            bins = np.bincount(encoded, minlength=num_classes * num_classes)
            confusion += bins.reshape(num_classes, num_classes)

        per_class_iou, miou = compute_per_class_iou(confusion, num_classes)
        print(f"\n{'=' * 50}")
        print(f"  mIoU: {miou:.4f}  ({miou * 100:.2f}%)")
        print(f"{'=' * 50}")

        import csv

        csv_path = output_dir / "per_class_iou.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["class_id", "iou"])
            for cls_id, iou_value in enumerate(per_class_iou):
                writer.writerow([cls_id, f"{iou_value:.6f}"])
        print(f"Per-class IoU saved to: {csv_path}")

        valid_classes = per_class_iou > 0
        class_ids = np.where(valid_classes)[0]
        iou_values = per_class_iou[valid_classes]

        fig, ax = plt.subplots(figsize=(max(12, len(class_ids) * 0.4), 5))
        ax.bar(class_ids, iou_values, color="steelblue")
        ax.axhline(miou, color="red", linestyle="--", label=f"mIoU={miou:.4f}")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("IoU")
        ax.set_title(f"Per-class IoU  (mIoU = {miou * 100:.2f}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        chart_path = output_dir / "per_class_iou_chart.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Per-class IoU chart saved to: {chart_path}")


if __name__ == "__main__":
    main()

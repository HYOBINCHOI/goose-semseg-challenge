import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch


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
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_miou": best_val_miou,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, save_path)


class RollingCheckpointManager:
    """Keeps only the top-K checkpoints ranked by mIoU (descending).

    Filenames encode both epoch and mIoU:
        top_miou_epoch_005_miou_0.4532.pt
    When a new checkpoint exceeds ``max_keep``, the lowest-scoring one
    is deleted automatically.
    """

    def __init__(self, run_dir: Path, max_keep: int = 2):
        self.run_dir = run_dir
        self.max_keep = max_keep
        # Each entry: (miou, epoch, path)
        self.entries: List[Tuple[float, int, Path]] = []
        # Recover existing rolling checkpoints on resume
        for p in sorted(run_dir.glob("top_miou_epoch_*_miou_*.pt")):
            try:
                parts = p.stem.split("_")
                ep = int(parts[3])
                miou = float(parts[5])
                self.entries.append((miou, ep, p))
            except (IndexError, ValueError):
                continue
        self.entries.sort(key=lambda x: x[0])

    def _build_path(self, epoch: int, miou: float) -> Path:
        return (self.run_dir /
                f"top_miou_epoch_{epoch + 1:03d}_miou_{miou:.4f}.pt")

    def should_save(self, miou: float) -> bool:
        if len(self.entries) < self.max_keep:
            return True
        return miou > self.entries[0][0]

    def save(
        self,
        model,
        optimizer,
        scaler,
        epoch: int,
        best_val_loss: float,
        best_val_miou: float,
        val_miou: float,
        args: argparse.Namespace,
    ) -> None:
        path = self._build_path(epoch, val_miou)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_loss,
            best_val_miou,
            args,
        )
        self.entries.append((val_miou, epoch, path))
        self.entries.sort(key=lambda x: x[0])

        while len(self.entries) > self.max_keep:
            _, _, evict_path = self.entries.pop(0)
            if evict_path.exists():
                evict_path.unlink()

    def summary(self) -> str:
        lines = []
        for miou, ep, p in reversed(self.entries):
            lines.append(f"  epoch {ep + 1:3d}  mIoU={miou:.4f}  {p.name}")
        return "\n".join(lines)


def ensure_epoch_log_file(log_path: Path) -> None:
    if log_path.exists():
        return

    with open(log_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_miou",
            "val_loss",
            "val_miou",
            "best_val_loss",
            "best_val_miou",
        ])


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
        writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{train_miou:.6f}",
            f"{val_loss:.6f}",
            f"{val_miou:.6f}",
            f"{best_val_loss:.6f}",
            f"{best_val_miou:.6f}",
        ])


def save_training_curves(log_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping training curve export.")
        return

    epochs = []
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []

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

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torchvision import transforms

from models import ConvNeXtMask2FormerBoostedModel


@dataclass
class LoadedCheckpointModel:
    backend_name: str
    checkpoint_args: dict
    crop: bool
    device: torch.device
    model: object
    num_classes: int
    payload: dict
    resize_size: Tuple[int, int]


def strip_file_prefix(path: Path | str) -> Path:
    return Path(str(path).removeprefix("file://"))


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Expected a Mask2Former training checkpoint with a model_state_dict field."
        )
    return payload


def resolve_checkpoint_backend(checkpoint_args: dict) -> str:
    model_type = str(checkpoint_args.get("model_type", "")).strip().lower()
    if model_type in {
        "",
        "convnext_mask2former",
        "convnext_mask2former_boosted",
    }:
        return "convnext_mask2former_boosted"

    if checkpoint_args.get("convnext_model_name_or_path"):
        return "convnext_mask2former_boosted"

    raise ValueError(
        "Unsupported checkpoint backend. Please add a backend branch for "
        f"model_type={model_type!r}."
    )


def _build_convnext_mask2former_boosted_model(
    payload: dict,
    device: torch.device,
) -> LoadedCheckpointModel:
    checkpoint_args = dict(payload.get("args", {}))
    checkpoint_args["device"] = str(device)

    args_namespace = argparse.Namespace(**checkpoint_args)
    num_classes = int(checkpoint_args.get("num_classes", 64))
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = ConvNeXtMask2FormerBoostedModel(args_namespace, id2label, label2id)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    resize_width = checkpoint_args.get("resize_width")
    resize_height = checkpoint_args.get("resize_height")
    if resize_width is None or resize_height is None:
        raise ValueError(
            "Checkpoint args must include resize_width and resize_height."
        )

    crop = bool(checkpoint_args.get("crop", False))
    if crop:
        raise ValueError(
            "Submission export currently supports checkpoints trained with crop=false only."
        )

    return LoadedCheckpointModel(
        backend_name="convnext_mask2former_boosted",
        checkpoint_args=checkpoint_args,
        crop=crop,
        device=device,
        model=model,
        num_classes=num_classes,
        payload=payload,
        resize_size=(int(resize_width), int(resize_height)),
    )


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedCheckpointModel:
    payload = load_checkpoint_payload(checkpoint_path)
    checkpoint_args = dict(payload.get("args", {}))
    backend_name = resolve_checkpoint_backend(checkpoint_args)

    if backend_name == "convnext_mask2former_boosted":
        return _build_convnext_mask2former_boosted_model(payload, device)

    raise ValueError(f"Unsupported checkpoint backend: {backend_name}")


def outputs_to_semantic_logits(
    outputs,
    target_size: Tuple[int, int],
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
    return semantic_logits


def run_model_logits(
    bundle: LoadedCheckpointModel,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    if bundle.backend_name == "convnext_mask2former_boosted":
        outputs = bundle.model(pixel_values=pixel_values)
        return outputs_to_semantic_logits(
            outputs, target_size=pixel_values.shape[-2:]
        )

    raise ValueError(f"Unsupported inference backend: {bundle.backend_name}")


def export_predictions_for_paths(
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    output_dir: Path,
    device_name: Optional[str] = None,
    output_name_resolver: Optional[Callable[[Path], str]] = None,
) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = build_model_from_checkpoint(checkpoint_path, device)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name_resolver is None:
        output_name_resolver = lambda path: path.name

    to_tensor = transforms.ToTensor()

    print(f"device={bundle.device}")
    print(f"checkpoint={checkpoint_path}")
    print(f"backend={bundle.backend_name}")
    print(f"crop={bundle.crop}")
    print(f"resize_size={bundle.resize_size}")
    print(f"prediction_image_count={len(image_paths)}")
    print(f"generated_predictions_dir={output_dir}")

    with torch.no_grad():
        for image_path in tqdm.tqdm(image_paths, desc="Generating predictions"):
            image = Image.open(image_path).convert("RGB")
            pixel_values = to_tensor(image).unsqueeze(0).to(bundle.device)

            semantic_logits = run_model_logits(bundle, pixel_values).to(torch.float32)
            prediction = (
                semantic_logits.argmax(dim=1)
                .squeeze(0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            prediction_np = np.asarray(prediction, dtype=np.uint8)
            destination_path = output_dir / output_name_resolver(image_path)
            Image.fromarray(prediction_np).save(destination_path)

    return output_dir

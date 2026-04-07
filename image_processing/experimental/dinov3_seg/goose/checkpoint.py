from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from goose.model import UpstreamDINOv3SegmentationModel


@dataclass
class LoadedExperimentalCheckpoint:
    checkpoint_args: dict
    device: torch.device
    model: UpstreamDINOv3SegmentationModel
    num_classes: int
    payload: dict


def strip_file_prefix(path: Union[Path, str]) -> Path:
    return Path(str(path).removeprefix("file://"))


def load_checkpoint_payload(checkpoint_path: Union[Path, str]) -> dict:
    checkpoint_path = strip_file_prefix(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Expected an experimental DINOv3 segmentation checkpoint with a model_state_dict field."
        )
    return payload


def build_model_from_checkpoint(
    checkpoint_path: Union[Path, str],
    device: torch.device,
) -> LoadedExperimentalCheckpoint:
    payload = load_checkpoint_payload(checkpoint_path)
    checkpoint_args = dict(payload.get("args", {}))
    checkpoint_args["device"] = str(device)

    args_namespace = argparse.Namespace(**checkpoint_args)
    num_classes = int(checkpoint_args.get("num_classes", 64))
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = UpstreamDINOv3SegmentationModel(args_namespace, id2label, label2id)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"],
        strict=False,
    )
    if missing:
        print(f"[INFO] Missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()

    return LoadedExperimentalCheckpoint(
        checkpoint_args=checkpoint_args,
        device=device,
        model=model,
        num_classes=num_classes,
        payload=payload,
    )

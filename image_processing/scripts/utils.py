import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

IMAGE_PROCESSING_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int) -> None:
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


def is_distributed_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args) -> Tuple[torch.device, int, int, int]:
    if not is_distributed_enabled():
        device = resolve_device(args.device)
        return device, 0, 0, 1

    if args.device != "cuda":
        raise ValueError(
            "Distributed training is only supported with --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for distributed training, but it is unavailable."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_goose_dataset_class(goose_tools_root: str):
    if goose_tools_root is None:
        goose_tools_root = str(IMAGE_PROCESSING_ROOT)
    goose_root = Path(goose_tools_root).resolve()
    if not goose_root.exists():
        raise FileNotFoundError(
            f"goose_tools_root does not exist: {goose_root}")

    if str(goose_root) not in sys.path:
        sys.path.insert(0, str(goose_root))

    from goosetools import GOOSE_Dataset

    return GOOSE_Dataset


def resolve_goose_data_root(data_path: str) -> Path:
    base_path = Path(data_path).expanduser().resolve()

    for root in (base_path, base_path / "goose-dataset"):
        if (root / "images" / "train").is_dir() and (root / "labels" /
                                                     "train").is_dir():
            return root

    raise FileNotFoundError(
        f"Could not find a valid GOOSE dataset root under {base_path}")


def default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scripts_dir = Path(__file__).resolve().parent
    return str(scripts_dir / "outputs" / timestamp)

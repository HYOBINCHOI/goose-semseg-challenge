from __future__ import annotations

import os
import sys
import traceback

from goose.args import parse_args
from goose.model import UpstreamDINOv3SegmentationModel
from goose.runtime import ensure_project_paths

ensure_project_paths()

from goose.train_utils import (build_dataloaders,  # noqa: E402
                               build_training_components, train_loop)
from utils import (cleanup_distributed, is_main_process,  # noqa: E402
                   load_goose_dataset_class, resolve_goose_data_root,
                   seed_everything, setup_distributed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device, local_rank, rank, world_size = setup_distributed(args)

    goose_dataset_class = load_goose_dataset_class(args.goose_tools_root)
    data_root = resolve_goose_data_root(args.data_path)

    train_dataset, val_dataset = goose_dataset_class.splits_from_path(
        str(data_root),
        resize_size=[args.resize_width, args.resize_height],
        crop=args.crop,
    )

    if is_main_process():
        print(f"Resolved GOOSE dataset root: {data_root}")
        print(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
        )
        if world_size > 1:
            print(
                "Distributed training enabled: "
                f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )
        print(
            "Using experimental upstream-style DINOv3 adapter + Mask2Former head "
            f"with interaction layers {args.vit_feature_indices}."
        )

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        args=args,
        device=device,
        world_size=world_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    id2label = {i: f"class_{i}" for i in range(args.num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model, optimizer, scaler, run_dir, epoch_log_path, curve_path = (
        build_training_components(
            args=args,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            model_class=UpstreamDINOv3SegmentationModel,
            id2label=id2label,
            label2id=label2id,
        ))

    try:
        train_loop(
            args=args,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            world_size=world_size,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            run_dir=run_dir,
            epoch_log_path=epoch_log_path,
            curve_path=curve_path,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(
            "[experimental/dinov3_seg/train.py] Unhandled exception on "
            f"rank={rank}, local_rank={local_rank}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise

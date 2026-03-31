import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from checkpoint_utils import (RollingCheckpointManager, append_epoch_log,
                              ensure_epoch_log_file, save_checkpoint,
                              save_training_curves)
from dataset_utils import GooseMask2FormerCollator, move_batch_to_device
from metrics import (compute_mean_iou, outputs_to_semantic_predictions,
                     update_confusion_matrix)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> AdamW:
    encoder_parameters = []
    other_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "pixel_level_module.encoder" in name:
            encoder_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []
    if other_parameters:
        parameter_groups.append({
            "params": other_parameters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        })
    if encoder_parameters:
        parameter_groups.append({
            "params": encoder_parameters,
            "lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
        })

    return AdamW(parameter_groups)


def maybe_enable_gradient_checkpointing(model: nn.Module) -> None:
    model_to_configure = model.module if hasattr(model, "module") else model

    if hasattr(model_to_configure, "gradient_checkpointing_enable"):
        try:
            model_to_configure.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on the top-level model.")
            return
        except ValueError as exc:
            print("Warning: top-level model does not support "
                  f"gradient checkpointing. Original error: {exc}")

    mask2former = getattr(model_to_configure, "mask2former", None)
    if mask2former is not None and hasattr(mask2former,
                                           "gradient_checkpointing_enable"):
        try:
            mask2former.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on Mask2Former.")
            return
        except ValueError as exc:
            print("Warning: Mask2Former does not support "
                  f"gradient checkpointing. Original error: {exc}")

    print(
        "Warning: gradient checkpointing was requested, but the model does not support it."
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: Optional[AdamW],
    use_amp: bool,
    grad_clip_norm: float,
    grad_accum_steps: int,
    num_classes: int,
    ignore_index: int,
    epoch_index: int,
    total_epochs: int,
    global_progress: Optional[tqdm],
) -> Tuple[float, float]:
    is_train = optimizer is not None
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1.")
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    confusion_matrix = torch.zeros((num_classes, num_classes),
                                   dtype=torch.int64,
                                   device=device)

    if is_main_process():
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=
            f"Epoch {epoch_index + 1}/{total_epochs} {'train' if is_train else 'val'}",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )
    else:
        progress_bar = loader

    if is_train:
        optimizer.zero_grad(set_to_none=True)

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
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = (step % grad_accum_steps == 0) or (step
                                                             == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        total_batches += 1

        predictions = outputs_to_semantic_predictions(
            outputs, target_size=batch["semantic_maps"].shape[-2:])
        update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            predictions=predictions,
            targets=batch["semantic_maps"],
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        if is_main_process():
            progress_bar.set_postfix(
                loss=f"{total_loss / step:.4f}",
                miou=f"{compute_mean_iou(confusion_matrix):.4f}",
            )

        if global_progress is not None:
            global_progress.update(1)
            global_progress.set_postfix(
                epoch=f"{epoch_index + 1}/{total_epochs}",
                phase="train" if is_train else "val",
                step=f"{step}/{len(loader)}",
                refresh=False,
            )

    if dist.is_available() and dist.is_initialized():
        count_tensor = torch.tensor(total_batches,
                                    dtype=torch.float64,
                                    device=device)
        loss_tensor = torch.tensor(total_loss,
                                   dtype=torch.float64,
                                   device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        mean_loss = float((loss_tensor / count_tensor.clamp_min(1.0)).item())
    else:
        mean_loss = total_loss / max(total_batches, 1)

    return mean_loss, compute_mean_iou(confusion_matrix)


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
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_miou = float(checkpoint.get("best_val_miou", float("-inf")))
    print(f"Resumed from epoch {start_epoch}.")
    return start_epoch, best_val_loss, best_val_miou


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
    world_size: int,
    train_dataset,
    val_dataset,
):
    collator = GooseMask2FormerCollator(ignore_index=args.ignore_index)
    pin_memory = device.type == "cuda" and not args.disable_pin_memory
    persistent_workers = args.persistent_workers and args.num_workers > 0
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False) if world_size > 1 and args.dist_eval else None

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "persistent_workers": persistent_workers,
    }
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "sampler": val_sampler,
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
    return train_loader, val_loader, train_sampler, val_sampler


def build_training_components(
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    world_size: int,
    model_class,
    id2label,
    label2id,
):
    if world_size > 1 and not is_main_process():
        dist.barrier()
    model = model_class(args, id2label, label2id).to(device)
    if world_size > 1 and is_main_process():
        dist.barrier()
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)

    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )

    run_dir = Path(args.output_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_path = run_dir / "epoch_metrics.csv"
    curve_path = run_dir / "training_curves.png"

    if is_main_process():
        with open(run_dir / "train_args.json", "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, indent=2)
        ensure_epoch_log_file(epoch_log_path)

    return model, optimizer, scaler, run_dir, epoch_log_path, curve_path


def train_loop(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    world_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler,
    val_sampler,
    run_dir: Path,
    epoch_log_path: Path,
    curve_path: Path,
) -> None:
    start_epoch, best_val_loss, best_val_miou = resume_if_needed(
        args, model, optimizer, scaler, device)

    total_training_steps = (args.epochs - start_epoch) * (len(train_loader) +
                                                          len(val_loader))
    global_progress = None
    if is_main_process():
        global_progress = tqdm(
            total=total_training_steps,
            desc="Total Progress",
            dynamic_ncols=True,
            position=0,
        )
    epochs_without_improvement = 0
    rolling_ckpt = RollingCheckpointManager(
        run_dir, max_keep=2) if is_main_process() else None

    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            train_loss, train_miou = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                scaler=scaler,
                optimizer=optimizer,
                use_amp=args.amp,
                grad_clip_norm=args.grad_clip_norm,
                grad_accum_steps=args.grad_accum_steps,
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
                grad_accum_steps=1,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                epoch_index=epoch,
                total_epochs=args.epochs,
                global_progress=global_progress,
            )

            if is_main_process():
                print(
                    f"epoch={epoch + 1} "
                    f"train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
                    f"val_loss={val_loss:.4f} val_miou={val_miou:.4f}")

            miou_improved = val_miou > (best_val_miou +
                                        args.early_stopping_min_delta)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if miou_improved:
                best_val_miou = val_miou
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if world_size > 1:
                loss_tensor = torch.tensor(best_val_loss,
                                           dtype=torch.float64,
                                           device=device)
                miou_tensor = torch.tensor(best_val_miou,
                                           dtype=torch.float64,
                                           device=device)
                dist.broadcast(loss_tensor, src=0)
                dist.broadcast(miou_tensor, src=0)
                best_val_loss = float(loss_tensor.item())
                best_val_miou = float(miou_tensor.item())

            if is_main_process():
                # Rolling top-2 mIoU checkpoints (+ latest = 3 files max)
                if rolling_ckpt.should_save(val_miou):
                    rolling_ckpt.save(
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        best_val_loss,
                        best_val_miou,
                        val_miou,
                        args,
                    )
                    print(
                        f"Saved top mIoU checkpoint (val_miou={val_miou:.4f}). "
                        f"Kept checkpoints:\n{rolling_ckpt.summary()}")

                # Always save latest (overwritten each epoch, for resume)
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

            if epochs_without_improvement >= args.early_stopping_patience:
                if is_main_process():
                    print("Early stopping triggered after "
                          f"{epochs_without_improvement} epochs "
                          "without val mIoU improvement.")
                break
    finally:
        if global_progress is not None:
            global_progress.close()

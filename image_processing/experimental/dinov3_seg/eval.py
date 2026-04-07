from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
import tqdm

from goose.checkpoint import build_model_from_checkpoint, strip_file_prefix
from goose.runtime import ensure_project_paths
from vendor.inference import make_inference

ensure_project_paths()

import evaluation as competition_eval  # noqa: E402
from goosetools import GOOSE_Dataset  # noqa: E402
from goosetools.data import load_splits  # noqa: E402
from goosetools.utils import str2bool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental DINOv3 adapter + upstream Mask2Former evaluation"
    )
    parser.add_argument("path", type=str, help="Path to GOOSE dataset root")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint to load")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Base output directory",
    )
    parser.add_argument("--iou", type=str2bool, default=True)
    parser.add_argument("--vis_res", type=str2bool, default=False)
    parser.add_argument("--n_classes", type=int, default=64)
    parser.add_argument("--test_split_name", type=str, default="val")
    parser.add_argument("--label_mapping_csv", type=str, default=None)
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["whole", "slide"],
        default="slide",
        help="Inference mode used by the vendored upstream helper.",
    )
    parser.add_argument(
        "--eval_crop_size",
        type=int,
        default=512,
        help="Crop size used for slide inference.",
    )
    parser.add_argument(
        "--eval_stride",
        type=int,
        default=341,
        help="Stride used for slide inference.",
    )
    parser.add_argument(
        "--num_max_forward",
        type=int,
        default=1,
        help="Padding forward passes for distributed slide inference parity.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Evaluation device. Falls back to cpu if cuda is unavailable.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_name)


def main() -> None:
    opt = parse_args()
    now = datetime.now()

    device = resolve_device(opt.device)
    print("Using device:", device)

    bundle = build_model_from_checkpoint(strip_file_prefix(opt.ckpt), device)

    n_classes = opt.n_classes
    if bundle.num_classes != n_classes:
        print(
            "[WARNING] Overriding --n_classes={} with checkpoint num_classes={}."
            .format(n_classes, bundle.num_classes)
        )
        n_classes = bundle.num_classes

    if opt.iou:
        mapping_csv = competition_eval.resolve_label_mapping_csv(
            opt.path, opt.label_mapping_csv
        )
        class_names, class_to_coarse = competition_eval.load_label_mapping(
            mapping_csv, n_classes
        )
    else:
        class_names = [str(i) for i in range(n_classes)]
        class_to_coarse = torch.full(
            (n_classes,),
            competition_eval.INVALID_COARSE_ID,
            dtype=torch.long,
        )

    validation_dict = load_splits(opt.path, [opt.test_split_name])[0]
    validation_dataset = GOOSE_Dataset(
        validation_dict,
        crop=False,
        resize_size=None,
        with_instances=False,
    )

    fine_conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    coarse_conf_mat = torch.zeros(
        (
            len(competition_eval.COARSE_CATEGORIES),
            len(competition_eval.COARSE_CATEGORIES) + 1,
        ),
        dtype=torch.int64,
    )

    try:
        print("*** Processing images ***")
        print("*************************")
        pbar = tqdm.tqdm(range(len(validation_dataset)))

        with torch.no_grad():
            for i in pbar:
                img, sem_map = validation_dataset[i]
                img_for_vis = img.clone()
                original_hw = tuple(int(x) for x in img.shape[-2:])

                semantic_logits = make_inference(
                    x=img.unsqueeze(0).to(bundle.device),
                    segmentation_model=bundle.model,
                    inference_mode=opt.inference_mode,
                    decoder_head_type="m2f",
                    rescale_to=original_hw,
                    n_output_channels=n_classes,
                    crop_size=(opt.eval_crop_size, opt.eval_crop_size),
                    stride=(opt.eval_stride, opt.eval_stride),
                    num_max_forward=opt.num_max_forward,
                )
                mask = semantic_logits.argmax(dim=1).squeeze(0).cpu().long()
                sem_map = sem_map.detach().cpu().long()

                if opt.vis_res:
                    competition_eval.visualize(img_for_vis.cpu(), sem_map, mask)

                if opt.iou:
                    competition_eval.update_fine_confusion(
                        fine_conf_mat, sem_map, mask, n_classes
                    )
                    competition_eval.update_coarse_confusion(
                        coarse_conf_mat, sem_map, mask, class_to_coarse, n_classes
                    )

                    if (i + 1) % 50 == 0 or (i + 1) == len(validation_dataset):
                        _, _, curr_fine = competition_eval.compute_fine_ious(
                            fine_conf_mat, n_classes
                        )
                        _, curr_coarse = competition_eval.compute_coarse_ious(
                            coarse_conf_mat
                        )
                        curr_comp = 0.5 * curr_fine + 0.5 * curr_coarse
                        pbar.set_postfix_str(
                            f"fine={curr_fine.item():.4f}, coarse={curr_coarse.item():.4f}, comp={curr_comp.item():.4f}"
                        )
    except KeyboardInterrupt:
        print("Interrupted by user, saving results until now.")
    except Exception as exc:
        print(f"An error occurred: {exc}")
        raise

    output_path = os.path.join(
        opt.output,
        "experimental_dinov3_seg_evaluation",
        now.strftime("%m-%d-%Y_%H-%M-%S"),
    )
    os.makedirs(output_path, exist_ok=True)

    if opt.iou:
        fine_class_ids, fine_ious, miou_fine = competition_eval.compute_fine_ious(
            fine_conf_mat, n_classes
        )
        coarse_ious, miou_coarse = competition_eval.compute_coarse_ious(
            coarse_conf_mat
        )
        miou_composite = 0.5 * miou_fine + 0.5 * miou_coarse

        competition_eval.write_results_txt_and_json(
            output_path=output_path,
            config=vars(opt),
            class_names=class_names,
            fine_class_ids=fine_class_ids,
            fine_ious=fine_ious,
            coarse_ious=coarse_ious,
            miou_fine=miou_fine,
            miou_coarse=miou_coarse,
            miou_composite=miou_composite,
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import super_gradients as sg
import torch
import tqdm
from goosetools import GOOSE_Dataset
from goosetools.data import load_splits
from goosetools.inference import run_inference
from goosetools.utils import str2bool
from matplotlib import pyplot as plt
from super_gradients.common.object_names import Models
from torchvision import transforms
from torchvision.transforms import InterpolationMode

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# -----------------------------
# Competition settings
# -----------------------------
EXCLUDED_FINE_CLASS_IDS = [0, 7, 9, 35, 44, 56, 61, 63]

COARSE_CATEGORIES = [
    "Animal",
    "Construction",
    "Human",
    "Object",
    "Road",
    "Sign",
    "Sky",
    "Terrain",
    "Vegetation",
    "Vehicle",
    "Water",
]
INVALID_COARSE_ID = len(COARSE_CATEGORIES)  # 11

DEFAULT_CLASSNAME_TO_COARSE = {
    "traffic_cone": "Sign",
    "snow": "Terrain",
    "cobble": "Road",
    "obstacle": "Object",
    "leaves": "Vegetation",
    "street_light": "Sign",
    "bikeway": "Road",
    "ego_vehicle": "Vehicle",
    "pedestrian_crossing": "Road",
    "road_block": "Object",
    "road_marking": "Road",
    "car": "Vehicle",
    "bicycle": "Vehicle",
    "person": "Human",
    "bus": "Vehicle",
    "forest": "Vegetation",
    "bush": "Vegetation",
    "moss": "Vegetation",
    "traffic_light": "Sign",
    "motorcycle": "Vehicle",
    "sidewalk": "Road",
    "curb": "Road",
    "asphalt": "Road",
    "gravel": "Terrain",
    "boom_barrier": "Sign",
    "rail_track": "Construction",
    "tree_crown": "Vegetation",
    "tree_trunk": "Vegetation",
    "debris": "Object",
    "crops": "Vegetation",
    "soil": "Terrain",
    "rider": "Human",
    "animal": "Animal",
    "truck": "Vehicle",
    "on_rails": "Vehicle",
    "caravan": "Vehicle",
    "trailer": "Vehicle",
    "building": "Construction",
    "wall": "Construction",
    "rock": "Terrain",
    "fence": "Construction",
    "guard_rail": "Construction",
    "bridge": "Construction",
    "tunnel": "Construction",
    "pole": "Construction",
    "traffic_sign": "Sign",
    "misc_sign": "Sign",
    "barrier_tape": "Sign",
    "kick_scooter": "Vehicle",
    "low_grass": "Vegetation",
    "high_grass": "Vegetation",
    "scenery_vegetation": "Vegetation",
    "sky": "Sky",
    "water": "Water",
    "wire": "Construction",
    "heavy_machinery": "Vehicle",
    "container": "Object",
    "hedge": "Vegetation",
    "barrel": "Object",
    "tree_root": "Vegetation",
}


def fill_coarse_mapping_from_class_names(
    class_names, class_to_coarse, coarse_name_to_id
) -> int:
    applied = 0
    for class_id, class_name in enumerate(class_names):
        if class_to_coarse[class_id] != INVALID_COARSE_ID:
            continue

        coarse_name = DEFAULT_CLASSNAME_TO_COARSE.get(class_name.strip().lower())
        if coarse_name is None:
            continue

        class_to_coarse[class_id] = coarse_name_to_id[coarse_name]
        applied += 1

    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="Path to GOOSE dataset root")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint to load")
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Path for output"
    )

    # Pre-processing
    parser.add_argument("--crop", action="store_true")
    parser.add_argument(
        "--use_processed_labels",
        action="store_true",
        help="Evaluate with processed labels (cropped/resized). "
             "For competition-like evaluation, keep this False.",
    )
    parser.add_argument("--resize_width", "-rw", type=int, default=768)
    parser.add_argument("--resize_height", "-rh", type=int, default=768)

    # Results
    parser.add_argument(
        "--iou",
        type=str2bool,
        default=True,
        help="Whether to calculate the competition scores or not. [Default True]",
    )
    parser.add_argument(
        "--vis_res",
        type=str2bool,
        default=False,
        help="Whether to visualize the results or not. [Default False]",
    )

    # Data
    parser.add_argument("--n_classes", "-nc", type=int, default=64)
    parser.add_argument(
        "--test_split_name",
        type=str,
        default="val",
        help="Split to evaluate on. Use val for local evaluation.",
    )
    parser.add_argument(
        "--label_mapping_csv",
        type=str,
        default=None,
        help="Path to goose_label_mapping.csv. "
             "If not set, {path}/goose_label_mapping.csv will be used.",
    )

    return parser.parse_args()


def strip_file_prefix(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://"):]
    return path


def nanmean_tensor(x: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(x)
    if valid.sum() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    return x[valid].mean()


def get_first_existing(row, candidates):
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


def resolve_label_mapping_csv(dataset_root: str, csv_path: str) -> str:
    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"label_mapping_csv not found: {csv_path}")
        return csv_path

    default_path = os.path.join(dataset_root, "goose_label_mapping.csv")
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            "Could not find goose_label_mapping.csv.\n"
            "Please pass it explicitly with --label_mapping_csv"
        )
    return default_path


def load_label_mapping(csv_path: str, n_classes: int):
    """
    Returns:
        class_names: list[str] of length n_classes
        class_to_coarse: torch.LongTensor of length n_classes
            - valid coarse ids: 0~10
            - INVALID_COARSE_ID(=11): ignored in coarse evaluation
    """
    class_names = [str(i) for i in range(n_classes)]
    class_to_coarse = [INVALID_COARSE_ID] * n_classes
    coarse_name_to_id = {name: idx for idx, name in enumerate(COARSE_CATEGORIES)}

    parsed_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            class_id_raw = get_first_existing(
                row, ["class_id", "id", "label_id", "train_id", "label_key"]
            )
            if class_id_raw is None:
                continue

            class_id = int(class_id_raw)
            if class_id < 0 or class_id >= n_classes:
                continue

            class_name_raw = get_first_existing(
                row, ["class_name", "name", "label_name"]
            )
            if class_name_raw is not None:
                class_names[class_id] = class_name_raw.strip()

            coarse_name_raw = get_first_existing(
                row,
                [
                    "category_name",
                    "category",
                    "superclass",
                    "super_class",
                    "coarse_category",
                    "coarse_name",
                ],
            )

            coarse_id_raw = get_first_existing(
                row,
                [
                    "category_id",
                    "coarse_id",
                    "superclass_id",
                    "super_class_id",
                ],
            )

            coarse_id = INVALID_COARSE_ID

            # Prefer category_name if available
            if coarse_name_raw is not None:
                coarse_name = coarse_name_raw.strip()
                if coarse_name.lower() != "void":
                    if coarse_name not in coarse_name_to_id:
                        raise ValueError(
                            "Unknown coarse category '{}' in {}\n"
                            "Expected one of: {}".format(
                                coarse_name, csv_path, COARSE_CATEGORIES
                            )
                        )
                    coarse_id = coarse_name_to_id[coarse_name]

            # Fallback to category_id if category_name is absent
            elif coarse_id_raw is not None:
                tmp_id = int(coarse_id_raw)
                # support both 0-based and 1-based ids
                if 0 <= tmp_id < len(COARSE_CATEGORIES):
                    coarse_id = tmp_id
                elif 1 <= tmp_id <= len(COARSE_CATEGORIES):
                    coarse_id = tmp_id - 1

            class_to_coarse[class_id] = coarse_id
            parsed_rows += 1

    if parsed_rows == 0:
        raise ValueError(
            "Failed to parse any rows from label mapping CSV: {}\n"
            "Please check the column names.".format(csv_path)
        )

    # If no coarse columns exist in CSV, all entries remain INVALID_COARSE_ID.
    # Fallback to built-in GOOSE class_name -> coarse mapping.
    if any(x == INVALID_COARSE_ID for x in class_to_coarse):
        applied = fill_coarse_mapping_from_class_names(
            class_names=class_names,
            class_to_coarse=class_to_coarse,
            coarse_name_to_id=coarse_name_to_id,
        )
        remaining = sum(x == INVALID_COARSE_ID for x in class_to_coarse)
        print(
            "[INFO] Applied {} fallback coarse mappings from class_name.".format(applied)
        )
        if remaining > 0:
            print(
                "[WARNING] {} classes still have no coarse mapping and will be ignored "
                "in coarse evaluation.".format(remaining)
            )

    return class_names, torch.tensor(class_to_coarse, dtype=torch.long)


def write_results_txt_and_json(
    output_path: str,
    config: dict,
    class_names,
    fine_class_ids,
    fine_ious,
    coarse_ious,
    miou_fine,
    miou_coarse,
    miou_composite,
):
    # text
    lines = []
    lines.append("Competition-style Evaluation")
    lines.append("=" * 60)
    lines.append("Excluded fine classes: {}".format(EXCLUDED_FINE_CLASS_IDS))
    lines.append("")
    lines.append("mIoUfine      : {}".format(miou_fine.item()))
    lines.append("mIoUcoarse    : {}".format(miou_coarse.item()))
    lines.append("mIoUcomposite : {}".format(miou_composite.item()))
    lines.append("")
    lines.append("[Fine IoU per class]")
    for idx, cls_id in enumerate(fine_class_ids):
        lines.append(
            "{:>2d} ({:<20s}) : {}".format(
                cls_id, class_names[cls_id], fine_ious[idx].item()
            )
        )

    lines.append("")
    lines.append("[Coarse IoU per category]")
    for idx, cat_name in enumerate(COARSE_CATEGORIES):
        lines.append("{:<12s} : {}".format(cat_name, coarse_ious[idx].item()))

    with open(os.path.join(output_path, "results.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            print(line)

    # json
    result_json = {
        "config": config,
        "excluded_fine_class_ids": EXCLUDED_FINE_CLASS_IDS,
        "mIoUfine": float(miou_fine.item()),
        "mIoUcoarse": float(miou_coarse.item()),
        "mIoUcomposite": float(miou_composite.item()),
        "fine_per_class": {
            str(cls_id): {
                "class_name": class_names[cls_id],
                "iou": None if torch.isnan(fine_ious[idx]) else float(fine_ious[idx].item()),
            }
            for idx, cls_id in enumerate(fine_class_ids)
        },
        "coarse_per_category": {
            COARSE_CATEGORIES[idx]: (
                None if torch.isnan(coarse_ious[idx]) else float(coarse_ious[idx].item())
            )
            for idx in range(len(COARSE_CATEGORIES))
        },
    }

    with open(os.path.join(output_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def visualize(img: torch.Tensor, gt: torch.Tensor, res: torch.Tensor):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img.permute(1, 2, 0).numpy())
    axes[0].axis("off")
    axes[0].set_title("RGB Image")

    axes[1].imshow(gt.numpy())
    axes[1].axis("off")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(res.numpy())
    axes[2].axis("off")
    axes[2].set_title("Inferred")

    plt.tight_layout()
    plt.show()


def update_fine_confusion(conf_mat, gt, pred, n_classes):
    """
    conf_mat: [n_classes, n_classes]
    Row = GT, Col = Pred
    GT pixels of excluded fine classes are ignored.
    """
    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)

    # ignore excluded GT classes
    for cls_id in EXCLUDED_FINE_CLASS_IDS:
        valid &= (gt != cls_id)

    gt = gt[valid]
    pred = pred[valid]

    if gt.numel() == 0:
        return

    indices = gt * n_classes + pred
    bincount = torch.bincount(indices, minlength=n_classes * n_classes)
    conf_mat += bincount.reshape(n_classes, n_classes)


def update_coarse_confusion(conf_mat, gt, pred, class_to_coarse, n_classes):
    """
    conf_mat: [11, 12]
      - GT coarse categories: 11 valid categories
      - Pred coarse categories: 11 valid + 1 invalid bucket
    GT pixels whose class maps to invalid/void are ignored.
    """
    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    gt = gt[valid]
    pred = pred[valid]

    if gt.numel() == 0:
        return

    gt_coarse = class_to_coarse[gt]
    pred_coarse = class_to_coarse[pred]

    # ignore GT that maps to invalid coarse category
    valid_gt = gt_coarse != INVALID_COARSE_ID
    gt_coarse = gt_coarse[valid_gt]
    pred_coarse = pred_coarse[valid_gt]

    if gt_coarse.numel() == 0:
        return

    num_pred_bins = INVALID_COARSE_ID + 1  # 12
    indices = gt_coarse * num_pred_bins + pred_coarse
    bincount = torch.bincount(
        indices, minlength=len(COARSE_CATEGORIES) * num_pred_bins
    )
    conf_mat += bincount.reshape(len(COARSE_CATEGORIES), num_pred_bins)


def compute_fine_ious(conf_mat, n_classes):
    conf_mat = conf_mat.to(torch.float64)

    tp = torch.diag(conf_mat)
    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat.sum(dim=0)

    fine_class_ids = [i for i in range(n_classes) if i not in EXCLUDED_FINE_CLASS_IDS]
    fine_ious = []

    for cls_id in fine_class_ids:
        union = row_sum[cls_id] + col_sum[cls_id] - tp[cls_id]
        if union > 0:
            fine_ious.append(tp[cls_id] / union)
        else:
            fine_ious.append(torch.tensor(float("nan"), dtype=torch.float64))

    fine_ious = torch.stack(fine_ious)
    miou_fine = nanmean_tensor(fine_ious)

    return fine_class_ids, fine_ious, miou_fine


def compute_coarse_ious(conf_mat):
    """
    conf_mat shape: [11, 12]
    """
    conf_mat = conf_mat.to(torch.float64)

    tp = conf_mat[:, :len(COARSE_CATEGORIES)].diag()
    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat[:, :len(COARSE_CATEGORIES)].sum(dim=0)

    coarse_ious = []
    for cat_id in range(len(COARSE_CATEGORIES)):
        union = row_sum[cat_id] + col_sum[cat_id] - tp[cat_id]
        if union > 0:
            coarse_ious.append(tp[cat_id] / union)
        else:
            coarse_ious.append(torch.tensor(float("nan"), dtype=torch.float64))

    coarse_ious = torch.stack(coarse_ious)
    miou_coarse = nanmean_tensor(coarse_ious)

    return coarse_ious, miou_coarse


def load_python_module(module_name: str, file_path: Path):
    scripts_dir = str(file_path.parent.resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def is_mask2former_checkpoint(ckpt_path: str) -> bool:
    return ckpt_path.endswith(".pt")


def load_mask2former_checkpoint_payload(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a Mask2Former training checkpoint with a model_state_dict field."
        )
    return checkpoint


def infer_mask2former_script_name(checkpoint_args: dict) -> str:
    run_name = str(checkpoint_args.get("run_name", ""))
    output_dir = str(checkpoint_args.get("output_dir", ""))

    if "convnext_mask2former_512_64_boosted" in run_name or "convnext_mask2former_512_64_boosted" in output_dir:
        return "convnext_mask2former_train_512_64_boosted.py"
    if "convnext_mask2former" in run_name or "convnext_mask2former" in output_dir:
        return "convnext_mask2former_train_512_64.py"
    if "dinov3_mask2former_regularized" in run_name or "dinov3_mask2former_regularized" in output_dir:
        return "dinov3_mask2former_train_regularized.py"
    if "dinov3" in run_name or "dinov3" in output_dir:
        return "dinov3_mask2former_train_512_64.py"
    raise ValueError(
        "Could not infer the training script from checkpoint args. "
        f"run_name={run_name}, output_dir={output_dir}"
    )


def build_mask2former_model_from_checkpoint(ckpt_path: str, device: torch.device):
    payload = load_mask2former_checkpoint_payload(ckpt_path)
    checkpoint_args = dict(payload.get("args", {}))
    checkpoint_args["device"] = str(device)

    script_name = infer_mask2former_script_name(checkpoint_args)
    script_path = TRAIN_SCRIPTS_DIR / script_name
    module = load_python_module(f"eval_{script_name.replace('.', '_')}", script_path)

    if hasattr(module, "ConvNeXtMask2FormerBoostedModel"):
        model_cls = module.ConvNeXtMask2FormerBoostedModel
    elif hasattr(module, "ConvNeXtMask2FormerModel"):
        model_cls = module.ConvNeXtMask2FormerModel
    elif hasattr(module, "DinoV3Mask2FormerModel"):
        model_cls = module.DinoV3Mask2FormerModel
    else:
        raise ValueError(f"Could not find a compatible model class in {script_path}")

    args_namespace = argparse.Namespace(**checkpoint_args)
    num_classes = int(checkpoint_args.get("num_classes", 64))
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = model_cls(args_namespace, id2label, label2id)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def run_mask2former_inference(img: torch.Tensor, model) -> torch.Tensor:
    mask2former_train = load_python_module(
        "eval_mask2former_train", TRAIN_SCRIPTS_DIR / "mask2former_train.py"
    )
    outputs = model(pixel_values=img.unsqueeze(0))
    prediction = mask2former_train.outputs_to_semantic_predictions(
        outputs, target_size=img.shape[-2:]
    )
    return prediction.squeeze(0)


if __name__ == "__main__":
    opt = parse_args()

    n_classes = opt.n_classes
    calculate_iou = opt.iou
    visualize_res = opt.vis_res
    now = datetime.now()

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load label mapping
    if calculate_iou:
        mapping_csv = resolve_label_mapping_csv(opt.path, opt.label_mapping_csv)
        class_names, class_to_coarse = load_label_mapping(mapping_csv, n_classes)
    else:
        class_names = [str(i) for i in range(n_classes)]
        class_to_coarse = torch.full((n_classes,), INVALID_COARSE_ID, dtype=torch.long)

    # Load model
    ckpt = strip_file_prefix(opt.ckpt)
    use_mask2former_checkpoint = is_mask2former_checkpoint(ckpt)
    if use_mask2former_checkpoint:
        model, payload = build_mask2former_model_from_checkpoint(ckpt, device)
        checkpoint_num_classes = int(payload["args"].get("num_classes", n_classes))
        if checkpoint_num_classes != n_classes:
            print(
                "[WARNING] Overriding --n_classes={} with checkpoint num_classes={}."
                .format(n_classes, checkpoint_num_classes)
            )
            n_classes = checkpoint_num_classes
    else:
        model = sg.training.models.get(
            model_name=Models.DDRNET_39,
            num_classes=n_classes,
            pretrained_weights=None,
            checkpoint_path=ckpt,
        )
        model = model.to(device)
        model.eval()

    # Load data
    validation_dict = load_splits(opt.path, [opt.test_split_name])[0]
    validation_dataset = GOOSE_Dataset(
        validation_dict,
        crop=opt.crop,
        resize_size=[opt.resize_width, opt.resize_height],
        with_instances=False,
    )

    # Competition-style confusion matrices
    fine_conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    coarse_conf_mat = torch.zeros(
        (len(COARSE_CATEGORIES), len(COARSE_CATEGORIES) + 1), dtype=torch.int64
    )

    try:
        print("*** Processing images ***")
        print("*************************")
        pbar = tqdm.tqdm(range(len(validation_dataset)))

        with torch.no_grad():
            for i in pbar:
                img, sem_map = validation_dataset[i]
                img_for_vis = img.clone()

                img = img.to(device)
                if use_mask2former_checkpoint:
                    mask = run_mask2former_inference(img, model=model)
                else:
                    mask = run_inference(img, model=model, threshold=0.5)

                if not opt.use_processed_labels:
                    sem_map = validation_dataset.get_original_label(i, True)
                    resize = transforms.Resize(
                        [sem_map.shape[0], sem_map.shape[1]],
                        interpolation=InterpolationMode.NEAREST,
                    )
                    mask = resize(mask.unsqueeze(0)).squeeze(0)

                mask = mask.detach().cpu().long()
                sem_map = sem_map.detach().cpu().long()

                if visualize_res:
                    visualize(img_for_vis.cpu(), sem_map, mask)

                if calculate_iou:
                    update_fine_confusion(fine_conf_mat, sem_map, mask, n_classes)
                    update_coarse_confusion(
                        coarse_conf_mat, sem_map, mask, class_to_coarse, n_classes
                    )

                    if (i + 1) % 50 == 0 or (i + 1) == len(validation_dataset):
                        _, _, curr_fine = compute_fine_ious(fine_conf_mat, n_classes)
                        _, curr_coarse = compute_coarse_ious(coarse_conf_mat)
                        curr_comp = 0.5 * curr_fine + 0.5 * curr_coarse
                        pbar.set_description(
                            "fine={:.4f}, coarse={:.4f}, comp={:.4f}".format(
                                curr_fine.item(), curr_coarse.item(), curr_comp.item()
                            )
                        )

    except KeyboardInterrupt:
        print("Interrupted by user, saving results until now.")
    except Exception as e:
        print("An error occurred: {}".format(e))
        raise

    output_path = os.path.join(
        opt.output, "evaluation", now.strftime("%m-%d-%Y_%H-%M-%S")
    )
    os.makedirs(output_path, exist_ok=True)

    if calculate_iou:
        fine_class_ids, fine_ious, miou_fine = compute_fine_ious(fine_conf_mat, n_classes)
        coarse_ious, miou_coarse = compute_coarse_ious(coarse_conf_mat)
        miou_composite = 0.5 * miou_fine + 0.5 * miou_coarse

        write_results_txt_and_json(
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

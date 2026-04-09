import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(TRAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS_DIR))

EXPERIMENTAL_DINOV3_DIR = PROJECT_ROOT / "experimental" / "dinov3_seg"
if str(EXPERIMENTAL_DINOV3_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_DINOV3_DIR))

import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from models import ConvNeXtMask2FormerBoostedModel
from goose.checkpoint import build_model_from_checkpoint as \
    build_experimental_model_from_checkpoint
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from make_submission_zip import (build_name_index, copy_submission_files,
                                 load_target_names, prepare_output_dir,
                                 validate_submission_zip, write_submission_zip)


def parse_args() -> argparse.Namespace:
    # 이 스크립트를 터미널에서 실행할 때 받을 옵션들을 정의한다.
    # 예를 들어 window 크기, stride, scale 목록, checkpoint 경로 등을
    # 코드 수정 없이 커맨드라인에서 바꿀 수 있게 해준다.
    parser = argparse.ArgumentParser(
        "Sliding-window + multi-scale inference for Mask2Former checkpoints")
    parser.add_argument("path", type=str, help="Path to GOOSE dataset root")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint to load")
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default="output",
                        help="Directory for outputs")
    parser.add_argument("--test_split_name",
                        type=str,
                        default="test",
                        help="Split to run inference on. Local validation is disabled, so use test.")
    parser.add_argument("--resize_width", type=int, default=512)
    parser.add_argument("--resize_height", type=int, default=512)
    parser.set_defaults(disable_resize=True)
    parser.add_argument(
        "--disable_resize",
        dest="disable_resize",
        action="store_true",
        help="Use raw image size at inference time instead of resizing first. "
        "This is the default behavior.",
    )
    parser.add_argument(
        "--enable_resize",
        dest="disable_resize",
        action="store_false",
        help="Resize before inference using --resize_width/--resize_height.",
    )
    parser.add_argument("--n_classes", type=int, default=64)
    # LOCAL VALIDATION DISABLED:
    # The arguments below were only used by the old val/evaluation path.
    # parser.add_argument(
    #     "--use_processed_labels",
    #     action="store_true",
    #     help="Evaluate with processed labels (cropped/resized). "
    #     "For competition-style evaluation, keep this False.",
    # )
    # parser.add_argument(
    #     "--label_mapping_csv",
    #     type=str,
    #     default=None,
    #     help="Path to goose_label_mapping.csv. "
    #     "If not set, {path}/goose_label_mapping.csv will be used.",
    # )
    parser.add_argument("--window_size",
                        type=int,
                        default=512,
                        help="Square crop size for sliding-window inference")
    parser.add_argument("--stride",
                        type=int,
                        default=384,
                        help="Sliding-window stride")
    parser.add_argument("--scales",
                        type=float,
                        nargs="+",
                        default=[1.0],
                        help="Inference scales, e.g. --scales 0.75 1.0 1.25")
    parser.add_argument("--disable_sliding_window",
                        action="store_true",
                        help="Run single-crop inference instead")
    parser.add_argument(
        "--merge_weighting",
        type=str,
        choices=("uniform", "center"),
        default="uniform",
        help="How to merge overlapping sliding-window crops. "
        "'uniform' reproduces the old behavior, while 'center' "
        "gives higher weight to crop centers.",
    )
    parser.add_argument(
        "--edge_weight_floor",
        type=float,
        default=0.1,
        help="Minimum edge weight when --merge_weighting center is used.",
    )
    parser.add_argument("--save_predictions",
                        action="store_true",
                        help="Save predicted masks as .pt tensors")
    parser.add_argument(
        "--save_prediction_pngs",
        action="store_true",
        help="Save prediction masks as PNG files. "
        "This is recommended for test-split submission generation.",
    )
    parser.add_argument(
        "--make_submission_zip",
        action="store_true",
        help="After test inference, filter generated PNGs and create a submission zip.",
    )
    parser.add_argument(
        "--scene_lists_dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "common"),
        help="Directory containing the official scene-list txt files.",
    )
    parser.add_argument(
        "--submission_output_dir",
        type=str,
        default=None,
        help="Directory where filtered submission PNGs will be copied.",
    )
    parser.add_argument(
        "--submission_output_zip",
        type=str,
        default=None,
        help="Path to the final submission zip file.",
    )
    parser.add_argument(
        "--expected_submission_count",
        type=int,
        default=361,
        help="Expected number of files in the final submission zip.",
    )
    return parser.parse_args()


def strip_file_prefix(path: str) -> str:
    # 입력 경로가 file://... 형태일 수 있어서,
    # 실제 파일 경로 문자열만 남기도록 접두어를 제거한다.
    return path.removeprefix("file://")


def load_checkpoint_payload(ckpt_path: str) -> dict:
    # 학습 중 저장된 checkpoint(.pt)를 로드한다.
    # 이 프로젝트의 checkpoint는 보통
    # - model_state_dict
    # - args
    # 같은 정보를 함께 들고 있으므로, 그 형식인지 확인한다.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a training checkpoint containing model_state_dict.")
    return checkpoint


def build_model_from_checkpoint(ckpt_path: str,
                                device: torch.device) -> Tuple[torch.nn.Module,
                                                                dict]:
    # checkpoint에 저장된 학습 당시 args를 바탕으로
    # 동일한 모델 구조를 다시 만든 뒤 가중치를 불러온다.
    #
    # 반환:
    # - model: 실제 추론에 사용할 모델
    # - payload: checkpoint 원본 dict (args 등 메타데이터 포함)
    payload = load_checkpoint_payload(ckpt_path)
    checkpoint_args = dict(payload.get("args", {}))
    state_dict_keys = payload.get("model_state_dict", {}).keys()

    # Experimental DINOv3 checkpoints use segmentation_backbone /
    # segmentation_head keys, while the original ConvNeXt Mask2Former
    # checkpoints use mask2former.* keys.
    is_experimental_dinov3 = any(
        key.startswith("segmentation_backbone.")
        or key.startswith("segmentation_head.")
        for key in state_dict_keys
    ) or bool(checkpoint_args.get("vit_feature_indices"))

    if is_experimental_dinov3:
        bundle = build_experimental_model_from_checkpoint(ckpt_path, device)
        print("[INFO] Loaded checkpoint backend: experimental_dinov3_seg")
        return bundle.model, bundle.payload

    checkpoint_args["device"] = str(device)
    args_namespace = argparse.Namespace(**checkpoint_args)

    num_classes = int(checkpoint_args.get("num_classes", 64))
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = ConvNeXtMask2FormerBoostedModel(args_namespace, id2label, label2id)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    print("[INFO] Loaded checkpoint backend: convnext_mask2former_boosted")
    return model, payload


def compute_semantic_logits(outputs,
                            target_size: Tuple[int, int]) -> torch.Tensor:
    # Mask2Former의 출력은 바로 픽셀별 클래스 맵이 아니라,
    # - query별 class logits
    # - query별 mask logits
    # 형태이다.
    #
    # 이 함수는 그 둘을 합쳐서
    # "클래스별 픽셀 점수 맵"인 semantic logits [B, C, H, W]로 변환한다.
    #
    # 반환값은 아직 argmax 이전 단계의 점수 맵이기 때문에,
    # sliding window / multiscale처럼 여러 결과를 평균낼 때 쓰기 좋다.
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


def is_experimental_dinov3_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "segmentation_backbone") and hasattr(
        model, "segmentation_head"
    ) and hasattr(model, "predict")


_PATCH_ALIGNMENT_LOGGED = False


def align_crop_for_model(model: torch.nn.Module,
                         crop: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # Experimental DINOv3 backbones assume patch-aligned spatial sizes.
    # In practice this adapter also uses CNN stages at /8, /16, /32, so
    # aligning to 32 keeps the token/grid bookkeeping consistent.
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        return crop, tuple(int(x) for x in crop.shape[-2:])

    patch_size = int(patch_size)
    alignment = max(patch_size, 32)
    original_height, original_width = (int(crop.shape[-2]), int(crop.shape[-1]))
    aligned_height = ((original_height + alignment - 1) // alignment) * alignment
    aligned_width = ((original_width + alignment - 1) // alignment) * alignment

    if (aligned_height, aligned_width) == (original_height, original_width):
        return crop, (original_height, original_width)

    global _PATCH_ALIGNMENT_LOGGED
    if not _PATCH_ALIGNMENT_LOGGED:
        print(
            "[INFO] Aligning crop sizes for patch-based backbone:",
            {
                "patch_size": patch_size,
                "alignment": alignment,
                "from": [original_height, original_width],
                "to": [aligned_height, aligned_width],
            },
        )
        _PATCH_ALIGNMENT_LOGGED = True

    aligned_crop = F.interpolate(
        crop,
        size=(aligned_height, aligned_width),
        mode="bilinear",
        align_corners=False,
    )
    return aligned_crop, (original_height, original_width)


def predict_crop_logits(model: torch.nn.Module,
                        crop: torch.Tensor) -> torch.Tensor:
    # 가장 기본이 되는 1회 추론 함수.
    #
    # 입력:
    # - crop: [1, 3, H, W] 형태의 이미지 한 장(또는 한 crop)
    #
    # 동작:
    # - 모델 forward 수행
    # - Mask2Former 출력을 semantic logits로 변환
    #
    # 반환:
    # - [1, C, H, W] 형태의 dense semantic logits
    #
    # sliding window는 결국 여러 crop에 대해 이 함수를 반복 호출하고,
    # 나온 logits를 원본 좌표계에 다시 합치는 방식으로 구현된다.
    aligned_crop, target_size = align_crop_for_model(model, crop)
    with torch.no_grad():
        if is_experimental_dinov3_model(model):
            # Match the baseline experimental DINOv3 inference path:
            # model.predict(...) upsamples pred_masks to rescale_to first, then
            # semantic logits are formed from pred_logits/pred_masks.
            predictions = model.predict(aligned_crop, rescale_to=target_size)
            mask_pred = predictions["pred_masks"]
            mask_cls = predictions["pred_logits"]
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            return torch.einsum(
                "bqc,bqhw->bchw",
                mask_cls.to(torch.float32),
                mask_pred.to(torch.float32),
            )

        outputs = model(pixel_values=aligned_crop)
    return compute_semantic_logits(outputs, target_size)


def generate_window_starts(length: int, window_size: int,
                           stride: int) -> List[int]:
    # 한 축(height 또는 width)에 대해 sliding window 시작 위치 목록을 만든다.
    #
    # 예:
    # length=1000, window=512, stride=384 라면
    # [0, 384, 488] 같은 식으로 마지막 window가 이미지 끝을 덮도록 맞춘다.
    #
    # 목적:
    # - 이미지 끝부분이 window 밖으로 버려지지 않게 하기 위함
    if length <= window_size:
        return [0]

    starts = list(range(0, length - window_size + 1, stride))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return starts


def build_merge_weight_map(height: int,
                           width: int,
                           device: torch.device,
                           mode: str,
                           edge_weight_floor: float) -> torch.Tensor:
    # 겹치는 crop를 합칠 때 사용할 2D 가중치 맵을 만든다.
    #
    # uniform:
    # - 기존 구현과 동일하게 모든 위치를 같은 가중치(1.0)로 본다.
    #
    # center:
    # - crop 중앙은 크게, 가장자리는 작게 가중치를 준다.
    # - 경계부 예측이 상대적으로 불안정할 수 있다는 가정하에
    #   중앙부 예측을 조금 더 신뢰하도록 만드는 방식이다.
    if mode == "uniform":
        return torch.ones((1, 1, height, width), device=device)

    edge_weight_floor = float(min(max(edge_weight_floor, 0.0), 1.0))
    y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device)

    # 가장자리에서는 0, 중앙에서는 1이 되는 1D 프로파일을 만든 뒤
    # floor를 더해 edge weight가 완전히 0이 되지 않게 한다.
    y_profile = 1.0 - y_coords.abs()
    x_profile = 1.0 - x_coords.abs()
    y_profile = edge_weight_floor + (1.0 - edge_weight_floor) * y_profile
    x_profile = edge_weight_floor + (1.0 - edge_weight_floor) * x_profile

    weight_map = torch.outer(y_profile, x_profile)
    return weight_map.unsqueeze(0).unsqueeze(0)


def sliding_window_inference(model: torch.nn.Module,
                             image: torch.Tensor,
                             num_classes: int,
                             window_size: int,
                             stride: int,
                             merge_weighting: str,
                             edge_weight_floor: float) -> torch.Tensor:
    # sliding window 추론의 핵심 함수.
    #
    # 입력 이미지를 여러 crop으로 나눠서 순차적으로 추론하고,
    # 각 crop의 semantic logits를 원본 크기 캔버스에 다시 누적한다.
    #
    # 겹치는 영역은 logits를 평균내어 하나의 최종 dense prediction으로 만든다.
    #
    # 왜 이렇게 하냐:
    # - 큰 이미지를 한 번에 넣으면 작은 객체/세부 경계가 약해질 수 있음
    # - crop 단위로 보면 상대적으로 더 자세한 정보를 볼 수 있음
    _, _, height, width = image.shape
    logits_sum = torch.zeros((1, num_classes, height, width),
                             device=image.device)
    logits_count = torch.zeros((1, 1, height, width), device=image.device)

    top_starts = generate_window_starts(height, window_size, stride)
    left_starts = generate_window_starts(width, window_size, stride)

    for top in top_starts:
        for left in left_starts:
            bottom = min(top + window_size, height)
            right = min(left + window_size, width)
            crop = image[:, :, top:bottom, left:right]
            crop_logits = predict_crop_logits(model, crop)
            crop_weight = build_merge_weight_map(
                height=bottom - top,
                width=right - left,
                device=image.device,
                mode=merge_weighting,
                edge_weight_floor=edge_weight_floor,
            )

            # crop 위치에 해당하는 원본 영역에 logits를 더한다.
            # 같은 위치가 여러 crop에 포함될 수 있으므로 count도 같이 누적한다.
            logits_sum[:, :, top:bottom, left:right] += crop_logits * crop_weight
            logits_count[:, :, top:bottom, left:right] += crop_weight

    # 누적된 logits를 count로 나눠, 겹친 영역은 평균 logits로 만든다.
    return logits_sum / logits_count.clamp_min(1.0)


def infer_single_scale(model: torch.nn.Module,
                       image: torch.Tensor,
                       num_classes: int,
                       use_sliding_window: bool,
                       window_size: int,
                       stride: int,
                       merge_weighting: str,
                       edge_weight_floor: float) -> torch.Tensor:
    # scale 하나에 대해서만 inference를 수행하는 wrapper 함수.
    #
    # - use_sliding_window=True  : window 단위로 나눠서 추론
    # - use_sliding_window=False : 이미지를 한 번에 넣어 추론
    #
    # multiscale 함수에서는 scale마다 이 함수를 호출한다.
    if use_sliding_window:
        return sliding_window_inference(model, image, num_classes, window_size,
                                        stride, merge_weighting,
                                        edge_weight_floor)
    return predict_crop_logits(model, image)


def multiscale_inference(model: torch.nn.Module,
                         image: torch.Tensor,
                         num_classes: int,
                         scales: Sequence[float],
                         use_sliding_window: bool,
                         window_size: int,
                         stride: int,
                         merge_weighting: str,
                         edge_weight_floor: float) -> torch.Tensor:
    # multiscale inference의 핵심 함수.
    #
    # 같은 이미지를 여러 scale로 resize한 뒤,
    # 각 scale에서 얻은 semantic logits를 다시 원본 크기로 맞춰서 평균낸다.
    #
    # 왜 이렇게 하냐:
    # - 작은 물체는 크게 볼 때 유리할 수 있고
    # - 큰 구조/문맥은 작게 볼 때 더 안정적일 수 있음
    # - 여러 scale의 예측을 합치면 더 robust한 결과를 얻을 수 있다.
    #
    # sliding window와 결합하면:
    # - 각 scale마다 sliding window 적용
    # - 그 결과를 scale 차원에서 다시 평균
    _, _, original_height, original_width = image.shape
    merged_logits = None

    for scale in scales:
        # 현재 scale에 맞춰 이미지 크기 변경
        scaled_height = max(int(round(original_height * scale)), 1)
        scaled_width = max(int(round(original_width * scale)), 1)
        scaled_image = F.interpolate(
            image,
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        )

        scaled_logits = infer_single_scale(
            model=model,
            image=scaled_image,
            num_classes=num_classes,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
            stride=stride,
            merge_weighting=merge_weighting,
            edge_weight_floor=edge_weight_floor,
        )

        # 서로 다른 scale 결과를 합치기 위해 원본 크기로 다시 올린다.
        if scaled_logits.shape[-2:] != (original_height, original_width):
            scaled_logits = F.interpolate(
                scaled_logits,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

        # scale별 logits 누적
        merged_logits = scaled_logits if merged_logits is None else merged_logits + scaled_logits

    # scale 개수만큼 평균
    return merged_logits / max(len(scales), 1)


def ensure_output_dir(output_root: str) -> Path:
    # 실행 결과를 타임스탬프별 폴더에 저장해
    # 여러 번 돌렸을 때 결과가 덮어써지지 않게 한다.
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    output_dir = Path(output_root) / "sliding_window_multiscale" / now
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_predictions_if_needed(output_dir: Path, dataset_index: int,
                               prediction: torch.Tensor) -> None:
    # 디버깅/시각화용으로 예측 마스크를 .pt로 저장한다.
    # 필요 없으면 --save_predictions 없이 실행하면 된다.
    save_path = output_dir / f"prediction_{dataset_index:05d}.pt"
    torch.save(prediction.cpu(), save_path)


def save_prediction_png(output_dir: Path, image_path: str,
                        prediction: torch.Tensor) -> None:
    # 제출용 예측은 PNG 마스크 파일로 저장한다.
    # make_submission_zip.py는 test prediction PNG 디렉토리를 입력으로 받는다.
    png_dir = output_dir / "prediction_pngs"
    png_dir.mkdir(parents=True, exist_ok=True)
    save_path = png_dir / Path(image_path).name
    Image.fromarray(prediction.cpu().numpy().astype("uint8"),
                    mode="L").save(save_path)


def collect_test_image_paths(dataset_root: str) -> List[str]:
    # test split은 라벨이 없으므로 이미지 경로를 직접 모은다.
    image_root = os.path.join(dataset_root, "images", "test")
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Test image directory not found: {image_root}")

    image_paths = sorted(glob.glob(os.path.join(image_root, "*", "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No test PNG files found under: {image_root}")
    return image_paths


def preprocess_image_like_dataset(image: Image.Image,
                                  resize_size: Optional[Sequence[int]]) -> torch.Tensor:
    # test image를 모델 입력 텐서로 바꾼다.
    # 기본은 raw-size inference이고, --enable_resize를 줬을 때만 resize한다.
    if resize_size is not None:
        image = image.resize(resize_size, resample=Image.BILINEAR)

    return transforms.ToTensor()(image)


def build_submission_from_predictions(
    prediction_png_dir: Path,
    scene_lists_dir: Path,
    submission_output_dir: Path,
    submission_output_zip: Path,
    expected_submission_count: Optional[int],
) -> Tuple[int, int]:
    target_names = load_target_names(
        scene_lists_dir,
        [
            "text file with ALICE scenes.txt",
            "text file with MuCAR-3 scenes.txt",
            "text file with Spotv1 scenes.txt",
            "text file with Spotv2 scenes.txt",
        ],
    )
    prediction_files = sorted(prediction_png_dir.rglob("*.png"))
    exact_base_map, prefix_timestamp_map = build_name_index(prediction_files)

    prepare_output_dir(submission_output_dir)
    created_files = copy_submission_files(
        target_names=target_names,
        output_dir=submission_output_dir,
        exact_base_map=exact_base_map,
        prefix_timestamp_map=prefix_timestamp_map,
    )
    write_submission_zip(submission_output_zip, created_files)
    validate_submission_zip(submission_output_zip, expected_submission_count)
    return len(target_names), len(created_files)


def main() -> None:
    # 전체 실행 흐름:
    # 1) 인자 파싱
    # 2) checkpoint에서 모델 복원
    # 3) test 데이터 로드
    # 4) sliding window + multiscale inference 수행
    # 5) prediction PNG 및 submission 산출물 저장
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) checkpoint로부터 학습된 모델 복원
    ckpt = strip_file_prefix(args.ckpt)
    model, payload = build_model_from_checkpoint(ckpt, device)
    checkpoint_num_classes = int(payload.get("args", {}).get("num_classes",
                                                             args.n_classes))
    if checkpoint_num_classes != args.n_classes:
        print(
            f"[WARNING] Overriding --n_classes={args.n_classes} with checkpoint "
            f"num_classes={checkpoint_num_classes}.")
    num_classes = checkpoint_num_classes

    is_test_split = args.test_split_name == "test"
    if not is_test_split:
        raise ValueError(
            "LOCAL VALIDATION DISABLED: this script is now test/submission "
            "only. Use --test_split_name test."
        )

    test_image_paths = collect_test_image_paths(args.path)
    output_dir = ensure_output_dir(args.output)

    with open(output_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    print("*** Running sliding-window + multiscale inference ***")
    print("*****************************************************")

    total_items = len(test_image_paths)
    pbar = tqdm.tqdm(range(total_items))
    logged_input_shape = False
    with torch.no_grad():
        for index in pbar:
            image_path = test_image_paths[index]
            original_image = Image.open(image_path).convert("RGB")
            original_height = original_image.height
            original_width = original_image.width
            image = preprocess_image_like_dataset(
                original_image,
                resize_size=None if args.disable_resize else
                [args.resize_width, args.resize_height],
            ).unsqueeze(0).to(device)

            if not logged_input_shape:
                print(
                    "First input sizes:",
                    {
                        "original_width": original_width,
                        "original_height": original_height,
                        "model_input_shape": list(image.shape),
                    },
                )
                logged_input_shape = True

            # 3) 현재 샘플에 대해 원하는 scale/window 설정으로 추론 수행
            logits = multiscale_inference(
                model=model,
                image=image,
                num_classes=num_classes,
                scales=args.scales,
                use_sliding_window=not args.disable_sliding_window,
                window_size=args.window_size,
                stride=args.stride,
                merge_weighting=args.merge_weighting,
                edge_weight_floor=args.edge_weight_floor,
            )

            # 4) dense logits -> 최종 semantic class map
            prediction = logits.argmax(dim=1).squeeze(0).cpu().long()

            resize = transforms.Resize(
                [original_height, original_width],
                interpolation=InterpolationMode.NEAREST,
            )
            prediction = resize(prediction.unsqueeze(0)).squeeze(0).long()
            pbar.set_description("test inference")

            if args.save_predictions:
                save_predictions_if_needed(output_dir, index, prediction)
            if args.save_prediction_pngs or is_test_split:
                save_prediction_png(output_dir, image_path, prediction)

    print("Finished test inference.")
    print(f"Prediction PNG directory: {output_dir / 'prediction_pngs'}")
    result_payload = {
        "mode": "test_inference",
        "num_images": total_items,
        "prediction_png_dir": str(output_dir / "prediction_pngs"),
        "num_classes": num_classes,
    }

    if args.make_submission_zip:
        submission_output_dir = Path(
            args.submission_output_dir) if args.submission_output_dir is not None else output_dir / "submission_pngs"
        submission_output_zip = Path(
            args.submission_output_zip) if args.submission_output_zip is not None else output_dir / "submission.zip"

        target_count, created_count = build_submission_from_predictions(
            prediction_png_dir=output_dir / "prediction_pngs",
            scene_lists_dir=Path(args.scene_lists_dir),
            submission_output_dir=submission_output_dir,
            submission_output_zip=submission_output_zip,
            expected_submission_count=args.expected_submission_count,
        )
        print(f"Submission PNG directory: {submission_output_dir}")
        print(f"Submission ZIP path: {submission_output_zip}")
        result_payload.update({
            "submission_target_count": target_count,
            "submission_created_count": created_count,
            "submission_output_dir": str(submission_output_dir),
            "submission_output_zip": str(submission_output_zip),
        })

    # LOCAL VALIDATION DISABLED:
    # The old val/evaluation path that computed fine/coarse/composite mIoU
    # was intentionally removed from execution flow. This script is now
    # submission/test inference only.
    with open(output_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(result_payload, fp, indent=2)


if __name__ == "__main__":
    main()

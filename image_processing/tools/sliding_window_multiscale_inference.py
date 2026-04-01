import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(TRAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS_DIR))

import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from goosetools import GOOSE_Dataset
from goosetools.data import load_splits
from models import ConvNeXtMask2FormerBoostedModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from evaluation import (compute_coarse_ious, compute_fine_ious,
                        load_label_mapping, resolve_label_mapping_csv,
                        update_coarse_confusion, update_fine_confusion)


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
                        default="val",
                        help="Split to evaluate on")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--resize_width", type=int, default=512)
    parser.add_argument("--resize_height", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=64)
    parser.add_argument(
        "--use_processed_labels",
        action="store_true",
        help="Evaluate with processed labels (cropped/resized). "
        "For competition-style evaluation, keep this False.",
    )
    parser.add_argument(
        "--label_mapping_csv",
        type=str,
        default=None,
        help="Path to goose_label_mapping.csv. "
        "If not set, {path}/goose_label_mapping.csv will be used.",
    )
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
    parser.add_argument("--save_predictions",
                        action="store_true",
                        help="Save predicted masks as .pt tensors")
    parser.add_argument(
        "--save_prediction_pngs",
        action="store_true",
        help="Save prediction masks as PNG files. "
        "This is recommended for test-split submission generation.",
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
    checkpoint = torch.load(ckpt_path, map_location="cpu")
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
    checkpoint_args["device"] = str(device)
    args_namespace = argparse.Namespace(**checkpoint_args)

    num_classes = int(checkpoint_args.get("num_classes", 64))
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    model = ConvNeXtMask2FormerBoostedModel(args_namespace, id2label, label2id)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
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
    with torch.no_grad():
        outputs = model(pixel_values=crop)
    return compute_semantic_logits(outputs, crop.shape[-2:])


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


def sliding_window_inference(model: torch.nn.Module,
                             image: torch.Tensor,
                             num_classes: int,
                             window_size: int,
                             stride: int) -> torch.Tensor:
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

            # crop 위치에 해당하는 원본 영역에 logits를 더한다.
            # 같은 위치가 여러 crop에 포함될 수 있으므로 count도 같이 누적한다.
            logits_sum[:, :, top:bottom, left:right] += crop_logits
            logits_count[:, :, top:bottom, left:right] += 1

    # 누적된 logits를 count로 나눠, 겹친 영역은 평균 logits로 만든다.
    return logits_sum / logits_count.clamp_min(1.0)


def infer_single_scale(model: torch.nn.Module,
                       image: torch.Tensor,
                       num_classes: int,
                       use_sliding_window: bool,
                       window_size: int,
                       stride: int) -> torch.Tensor:
    # scale 하나에 대해서만 inference를 수행하는 wrapper 함수.
    #
    # - use_sliding_window=True  : window 단위로 나눠서 추론
    # - use_sliding_window=False : 이미지를 한 번에 넣어 추론
    #
    # multiscale 함수에서는 scale마다 이 함수를 호출한다.
    if use_sliding_window:
        return sliding_window_inference(model, image, num_classes, window_size,
                                        stride)
    return predict_crop_logits(model, image)


def multiscale_inference(model: torch.nn.Module,
                         image: torch.Tensor,
                         num_classes: int,
                         scales: Sequence[float],
                         use_sliding_window: bool,
                         window_size: int,
                         stride: int) -> torch.Tensor:
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


def preprocess_image_like_dataset(image: Image.Image, crop: bool,
                                  resize_size: Sequence[int]) -> torch.Tensor:
    # GOOSE_Dataset.preprocess와 동일한 규칙으로 test image를 전처리한다.
    if crop:
        crop_ratio = resize_size[0] / resize_size[1]
        input_ratio = image.width / image.height

        if input_ratio > crop_ratio:
            new_height = image.height
            new_width = int(new_height * crop_ratio)
        elif crop_ratio > input_ratio:
            new_width = image.width
            new_height = int(new_width // crop_ratio)
        else:
            new_width = image.width
            new_height = image.height

        image = transforms.CenterCrop((new_height, new_width)).forward(image)

    if resize_size is not None:
        image = image.resize(resize_size, resample=Image.BILINEAR)

    return transforms.ToTensor()(image)


def main() -> None:
    # 전체 실행 흐름:
    # 1) 인자 파싱
    # 2) checkpoint에서 모델 복원
    # 3) 평가 데이터셋 로드
    # 4) sliding window + multiscale inference 수행
    # 5) prediction으로 mIoU 계산 및 결과 저장
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

    mapping_csv = resolve_label_mapping_csv(args.path, args.label_mapping_csv)
    _, class_to_coarse = load_label_mapping(mapping_csv, num_classes)

    is_test_split = args.test_split_name == "test"

    if is_test_split:
        test_image_paths = collect_test_image_paths(args.path)
        validation_dataset = None
    else:
        # 2) 평가할 split 로드
        validation_dict = load_splits(args.path, [args.test_split_name])[0]
        validation_dataset = GOOSE_Dataset(
            validation_dict,
            crop=args.crop,
            resize_size=[args.resize_width, args.resize_height],
            with_instances=False,
        )

    # evaluation.py와 같은 competition-style confusion matrix를 사용한다.
    fine_conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    coarse_conf_mat = torch.zeros((11, 12), dtype=torch.int64)
    output_dir = ensure_output_dir(args.output)

    with open(output_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    print("*** Running sliding-window + multiscale inference ***")
    print("*****************************************************")

    total_items = len(test_image_paths) if is_test_split else len(
        validation_dataset)
    pbar = tqdm.tqdm(range(total_items))
    with torch.no_grad():
        for index in pbar:
            if is_test_split:
                image_path = test_image_paths[index]
                original_image = Image.open(image_path).convert("RGB")
                original_height = original_image.height
                original_width = original_image.width
                image = preprocess_image_like_dataset(
                    original_image,
                    crop=args.crop,
                    resize_size=[args.resize_width, args.resize_height],
                ).unsqueeze(0).to(device)
                semantic_map = None
            else:
                image, semantic_map = validation_dataset[index]
                image_path = validation_dataset.dataset_dict[index]["img_path"]
                image = image.unsqueeze(0).to(device)
                original_height = None
                original_width = None

            # 3) 현재 샘플에 대해 원하는 scale/window 설정으로 추론 수행
            logits = multiscale_inference(
                model=model,
                image=image,
                num_classes=num_classes,
                scales=args.scales,
                use_sliding_window=not args.disable_sliding_window,
                window_size=args.window_size,
                stride=args.stride,
            )

            # 4) dense logits -> 최종 semantic class map
            prediction = logits.argmax(dim=1).squeeze(0).cpu().long()

            if is_test_split:
                resize = transforms.Resize(
                    [original_height, original_width],
                    interpolation=InterpolationMode.NEAREST,
                )
                prediction = resize(prediction.unsqueeze(0)).squeeze(0).long()
            else:
                # competition-style 평가는 원본 라벨 해상도 기준으로 계산한다.
                if not args.use_processed_labels:
                    semantic_map = validation_dataset.get_original_label(index, True)

                # GT label 크기와 다르면 nearest 보간으로 prediction 크기 맞춤
                if semantic_map.shape[-2:] != prediction.shape[-2:]:
                    resize = transforms.Resize(
                        [semantic_map.shape[0], semantic_map.shape[1]],
                        interpolation=InterpolationMode.NEAREST,
                    )
                    prediction = resize(prediction.unsqueeze(0)).squeeze(0).long()

                semantic_map = semantic_map.cpu().long()

                # evaluation.py와 같은 fine/coarse confusion 업데이트
                update_fine_confusion(fine_conf_mat, semantic_map, prediction,
                                      num_classes)
                update_coarse_confusion(coarse_conf_mat, semantic_map, prediction,
                                        class_to_coarse, num_classes)

                _, _, current_fine = compute_fine_ious(fine_conf_mat,
                                                       num_classes)
                _, current_coarse = compute_coarse_ious(coarse_conf_mat)
                current_comp = 0.5 * current_fine + 0.5 * current_coarse
                pbar.set_description(
                    f"fine={current_fine.item():.4f}, "
                    f"coarse={current_coarse.item():.4f}, "
                    f"comp={current_comp.item():.4f}")
            if is_test_split:
                pbar.set_description("test inference")

            if args.save_predictions:
                save_predictions_if_needed(output_dir, index, prediction)
            if args.save_prediction_pngs or is_test_split:
                save_prediction_png(output_dir, image_path, prediction)

    if is_test_split:
        print("Finished test inference.")
        print(f"Prediction PNG directory: {output_dir / 'prediction_pngs'}")
        with open(output_dir / "results.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "mode": "test_inference",
                    "num_images": total_items,
                    "prediction_png_dir": str(output_dir / "prediction_pngs"),
                    "num_classes": num_classes,
                },
                fp,
                indent=2,
            )
    else:
        _, fine_ious, final_fine = compute_fine_ious(fine_conf_mat,
                                                     num_classes)
        coarse_ious, final_coarse = compute_coarse_ious(coarse_conf_mat)
        final_composite = 0.5 * final_fine + 0.5 * final_coarse

        print(f"Final mIoU fine: {final_fine.item():.6f}")
        print(f"Final mIoU coarse: {final_coarse.item():.6f}")
        print(f"Final mIoU composite: {final_composite.item():.6f}")

        with open(output_dir / "results.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "mIoU_fine": float(final_fine.item()),
                    "mIoU_coarse": float(final_coarse.item()),
                    "mIoU_composite": float(final_composite.item()),
                    "num_classes": num_classes,
                    "num_fine_classes_with_union": int(
                        torch.sum(~torch.isnan(fine_ious)).item()),
                    "num_coarse_categories_with_union": int(
                        torch.sum(~torch.isnan(coarse_ious)).item()),
                },
                fp,
                indent=2,
            )


if __name__ == "__main__":
    main()

#!/opt/conda/envs/goose/bin/python
"""
Create a Codabench submission zip from a Mask2Former training checkpoint (.pt),
end-to-end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from goosetools.checkpoint_inference import (
    export_predictions_for_paths,
    strip_file_prefix,
)


SENSOR_SUFFIXES: Tuple[str, ...] = (
    "_camera_left",
    "_windshield_vis",
    "_front",
    "_realsense",
)

DEFAULT_LIST_FILES: Tuple[str, ...] = (
    "text file with ALICE scenes.txt",
    "text file with MuCAR-3 scenes.txt",
    "text file with Spotv1 scenes.txt",
    "text file with Spotv2 scenes.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Codabench submission zip directly from a Mask2Former "
            "checkpoint."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to the GOOSE dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Mask2Former training checkpoint (.pt).",
    )
    parser.add_argument(
        "--generated_predictions_dir",
        type=Path,
        default=None,
        help=(
            "Directory where generated prediction PNGs will be written before "
            "building the submission zip."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to export when using --checkpoint. Default: test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--scene_lists_dir",
        type=Path,
        default=REPO_ROOT / "common",
        help=(
            "Directory that contains the official txt files listing target scenes. "
            "Default: <repo>/common"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the filtered submission PNGs will be written.",
    )
    parser.add_argument(
        "--output_zip",
        type=Path,
        required=True,
        help="Path to the final submission zip file.",
    )
    parser.add_argument(
        "--list_files",
        nargs="+",
        default=list(DEFAULT_LIST_FILES),
        help="Txt files to read from --scene_lists_dir.",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=None,
        help="Optional expected number of submission files. Default: target count.",
    )
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    checkpoint_path = strip_file_prefix(args.checkpoint)
    if checkpoint_path.suffix != ".pt":
        raise ValueError(
            "Checkpoint inference currently supports Mask2Former training checkpoints (.pt) only."
        )
    if args.generated_predictions_dir is None:
        args.generated_predictions_dir = args.output_dir.parent / (
            args.output_dir.name + "_generated_predictions"
        )

    if args.generated_predictions_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "--generated_predictions_dir must be different from --output_dir."
        )


def strip_sensor_suffix(stem: str) -> str:
    for suffix in SENSOR_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def strip_labelids_suffix(stem: str) -> str:
    suffix = "_labelids"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def load_target_names(scene_lists_dir: Path, list_files: Sequence[str]) -> List[str]:
    target_names: List[str] = []
    for filename in list_files:
        path = scene_lists_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing scene list file: {path}")
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    target_names.append(line)
    return list(dict.fromkeys(target_names))


def build_name_index(
    paths: Iterable[Path],
) -> Tuple[Dict[str, Path], Dict[Tuple[str, str], Path]]:
    exact_base_map: Dict[str, Path] = {}
    prefix_timestamp_map: Dict[Tuple[str, str], Path] = {}

    for path in paths:
        # Normalize both raw image names (sensor suffix) and generated
        # submission names (_labelids suffix) to the same base identifier.
        base = strip_labelids_suffix(strip_sensor_suffix(path.stem))
        exact_base_map[base] = path

        try:
            prefix, _frame_idx, timestamp = base.rsplit("_", 2)
        except ValueError:
            continue
        prefix_timestamp_map[(prefix, timestamp)] = path

    return exact_base_map, prefix_timestamp_map


def resolve_target_path(
    target_name: str,
    exact_base_map: Dict[str, Path],
    prefix_timestamp_map: Dict[Tuple[str, str], Path],
    missing_message_prefix: str,
) -> Path:
    target_base = strip_labelids_suffix(Path(target_name).stem)
    source = exact_base_map.get(target_base)
    if source is not None:
        return source

    try:
        prefix, _frame_idx, timestamp = target_base.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Could not parse target filename: {target_name}") from exc

    source = prefix_timestamp_map.get((prefix, timestamp))
    if source is None:
        raise FileNotFoundError(f"{missing_message_prefix}: {target_name}")
    return source


def collect_prediction_files(
    predictions_dir: Path,
) -> List[Path]:
    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory does not exist: {predictions_dir}"
        )

    prediction_files = list(predictions_dir.rglob("*.png"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction PNGs found under: {predictions_dir}")
    return prediction_files


def collect_image_paths(dataset_root: Path, split: str) -> List[Path]:
    split_root = dataset_root / "images" / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")

    image_paths = sorted(split_root.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under: {split_root}")
    return image_paths


def select_image_paths_for_targets(
    image_paths: Sequence[Path],
    target_names: Sequence[str],
) -> List[Path]:
    exact_base_map, prefix_timestamp_map = build_name_index(image_paths)
    selected_paths = [
        resolve_target_path(
            target_name=target_name,
            exact_base_map=exact_base_map,
            prefix_timestamp_map=prefix_timestamp_map,
            missing_message_prefix="No test image matched target",
        )
        for target_name in target_names
    ]
    return list(dict.fromkeys(selected_paths))


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_submission_files(
    target_names: Sequence[str],
    output_dir: Path,
    exact_base_map: Dict[str, Path],
    prefix_timestamp_map: Dict[Tuple[str, str], Path],
) -> List[Path]:
    created_files: List[Path] = []
    missing_targets: List[str] = []

    for target_name in target_names:
        try:
            source = resolve_target_path(
                target_name=target_name,
                exact_base_map=exact_base_map,
                prefix_timestamp_map=prefix_timestamp_map,
                missing_message_prefix="No prediction matched target",
            )
        except FileNotFoundError:
            missing_targets.append(target_name)
            continue

        destination = output_dir / target_name
        shutil.copy2(source, destination)
        created_files.append(destination)

    if missing_targets:
        preview = ", ".join(missing_targets[:10])
        extra = "" if len(missing_targets) <= 10 else f" ... (+{len(missing_targets) - 10})"
        raise FileNotFoundError(
            f"Missing predictions for {len(missing_targets)} targets: {preview}{extra}"
        )

    return created_files


def write_submission_zip(output_zip: Path, created_files: Sequence[Path]) -> None:
    if output_zip.exists():
        output_zip.unlink()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path in sorted(created_files):
            zf.write(path, arcname=path.name)


def validate_submission_zip(output_zip: Path, expected_count: Optional[int]) -> None:
    with zipfile.ZipFile(output_zip) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        png_names = [name for name in names if name.lower().endswith(".png")]
        has_subdirs = any("/" in name for name in names)

    if has_subdirs:
        raise ValueError(
            "Submission zip contains subdirectories; expected root-level PNGs only."
        )
    if len(names) != len(png_names):
        raise ValueError("Submission zip contains non-PNG files.")
    if expected_count is not None and len(png_names) != expected_count:
        raise ValueError(
            f"Submission zip contains {len(png_names)} PNGs, expected {expected_count}."
        )


def build_prediction_output_name(image_path: Path) -> str:
    return strip_sensor_suffix(image_path.stem) + "_labelids.png"


def main() -> None:
    args = parse_args()
    validate_args(args)

    target_names = load_target_names(args.scene_lists_dir, args.list_files)
    expected_count = args.expected_count if args.expected_count is not None else len(target_names)

    dataset_root = strip_file_prefix(args.dataset_root)
    image_paths = collect_image_paths(dataset_root, args.split)
    image_paths = select_image_paths_for_targets(image_paths, target_names)

    # Reuse already generated PNGs when they exist. This avoids re-running
    # model inference and allows submission packaging for backends that are
    # not yet wired into goosetools/checkpoint_inference.py.
    existing_prediction_files = []
    if args.generated_predictions_dir.exists():
        existing_prediction_files = list(
            args.generated_predictions_dir.rglob("*.png")
        )

    if existing_prediction_files:
        predictions_dir = args.generated_predictions_dir
        print(
            "Using existing generated predictions:",
            predictions_dir,
        )
    else:
        predictions_dir = export_predictions_for_paths(
            image_paths=image_paths,
            checkpoint_path=strip_file_prefix(args.checkpoint),
            output_dir=args.generated_predictions_dir,
            device_name=args.device,
            output_name_resolver=build_prediction_output_name,
        )

    prediction_files = collect_prediction_files(
        predictions_dir=predictions_dir,
    )
    exact_base_map, prefix_timestamp_map = build_name_index(prediction_files)

    prepare_output_dir(args.output_dir)
    created_files = copy_submission_files(
        target_names=target_names,
        output_dir=args.output_dir,
        exact_base_map=exact_base_map,
        prefix_timestamp_map=prefix_timestamp_map,
    )
    write_submission_zip(args.output_zip, created_files)
    validate_submission_zip(args.output_zip, expected_count)

    print(f"target_count={len(target_names)}")
    print(f"created_count={len(created_files)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_zip={args.output_zip}")


if __name__ == "__main__":
    main()

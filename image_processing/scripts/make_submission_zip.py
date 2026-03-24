#!/opt/conda/envs/goose/bin/python
"""
Create a Codabench submission zip from full-test semantic predictions.

"""

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
        description="Build a Codabench submission zip from GOOSE prediction PNGs."
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        required=True,
        help="Directory containing full-test prediction PNGs.",
    )
    parser.add_argument(
        "--scene_lists_dir",
        type=Path,
        required=True,
        help="Directory that contains the official txt files listing target scenes.",
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
        help="Optional expected number of submission files, e.g. 361.",
    )
    parser.add_argument(
        "--sample_check_dir_name",
        type=str,
        default="sample_check",
        help="Prediction subdirectory name to ignore when scanning PNGs.",
    )
    return parser.parse_args()


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


def collect_prediction_files(
    predictions_dir: Path, ignored_dir_name: str
) -> List[Path]:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    prediction_files = [
        path
        for path in predictions_dir.rglob("*.png")
        if ignored_dir_name not in path.parts
    ]
    if not prediction_files:
        raise FileNotFoundError(f"No prediction PNGs found under: {predictions_dir}")
    return prediction_files


def build_prediction_index(
    prediction_files: Iterable[Path],
) -> Tuple[Dict[str, Path], Dict[Tuple[str, str], Path]]:
    exact_base_map: Dict[str, Path] = {}
    prefix_timestamp_map: Dict[Tuple[str, str], Path] = {}

    for path in prediction_files:
        base = strip_sensor_suffix(path.stem)
        exact_base_map[base] = path

        try:
            prefix, _frame_idx, timestamp = base.rsplit("_", 2)
        except ValueError:
            continue
        prefix_timestamp_map[(prefix, timestamp)] = path

    return exact_base_map, prefix_timestamp_map


def resolve_prediction_path(
    target_name: str,
    exact_base_map: Dict[str, Path],
    prefix_timestamp_map: Dict[Tuple[str, str], Path],
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
        raise FileNotFoundError(f"No prediction matched target: {target_name}")
    return source


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
            source = resolve_prediction_path(
                target_name=target_name,
                exact_base_map=exact_base_map,
                prefix_timestamp_map=prefix_timestamp_map,
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

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(created_files):
            zf.write(path, arcname=path.name)


def validate_submission_zip(output_zip: Path, expected_count: Optional[int]) -> None:
    with zipfile.ZipFile(output_zip) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        png_names = [name for name in names if name.lower().endswith(".png")]
        has_subdirs = any("/" in name for name in names)

    if has_subdirs:
        raise ValueError("Submission zip contains subdirectories; expected root-level PNGs only.")
    if len(names) != len(png_names):
        raise ValueError("Submission zip contains non-PNG files.")
    if expected_count is not None and len(png_names) != expected_count:
        raise ValueError(
            f"Submission zip contains {len(png_names)} PNGs, expected {expected_count}."
        )


def main() -> None:
    args = parse_args()

    target_names = load_target_names(args.scene_lists_dir, args.list_files)
    prediction_files = collect_prediction_files(
        predictions_dir=args.predictions_dir,
        ignored_dir_name=args.sample_check_dir_name,
    )
    exact_base_map, prefix_timestamp_map = build_prediction_index(prediction_files)

    prepare_output_dir(args.output_dir)
    created_files = copy_submission_files(
        target_names=target_names,
        output_dir=args.output_dir,
        exact_base_map=exact_base_map,
        prefix_timestamp_map=prefix_timestamp_map,
    )
    write_submission_zip(args.output_zip, created_files)
    validate_submission_zip(args.output_zip, args.expected_count)

    print(f"target_count={len(target_names)}")
    print(f"created_count={len(created_files)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_zip={args.output_zip}")


if __name__ == "__main__":
    main()
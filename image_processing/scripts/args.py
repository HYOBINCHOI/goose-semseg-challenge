import argparse
import json
from pathlib import Path
from typing import Dict

from utils import IMAGE_PROCESSING_ROOT, default_output_dir

DEFAULT_GOOSE_TOOLS_ROOT = str(IMAGE_PROCESSING_ROOT)
DEFAULT_CONFIG_PATH = IMAGE_PROCESSING_ROOT / "config" / "train.yaml"


def load_config_file(config_path: str) -> Dict[str, object]:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_file}")

    suffix = config_file.suffix.lower()
    with open(config_file, "r", encoding="utf-8") as fp:
        if suffix == ".json":
            data = json.load(fp)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML configs. Install with `pip install pyyaml`."
                ) from exc
            data = yaml.safe_load(fp)
        else:
            raise ValueError(
                f"Unsupported config format: {config_file.suffix}. Use .yaml, .yml, or .json."
            )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            "Config file must contain a top-level mapping/object.")
    return data


def validate_config_keys(parser: argparse.ArgumentParser,
                         config_values: Dict[str, object]) -> None:
    valid_keys = {
        action.dest
        for action in parser._actions if action.dest != argparse.SUPPRESS
    }
    unknown_keys = sorted(set(config_values) - valid_keys)
    if unknown_keys:
        raise ValueError("Unknown config keys: " + ", ".join(unknown_keys))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "ConvNeXt + Mask2Former Trainer (boosted)")
    gradient_checkpointing_help = "Enable gradient checkpointing when supported by the loaded model."
    convnext_help = (
        "Pretrained ConvNeXt backbone checkpoint. Use a public Hugging Face model "
        "unless you have access to gated repos.")
    mask2former_help = "Pretrained Mask2Former checkpoint for decoder/heads initialization."
    freeze_decoder_help = "Freeze pretrained Mask2Former transformer decoder and prediction heads."

    parser.add_argument("data_path",
                        nargs="?",
                        default=None,
                        help="Path to goose dataset root")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=
        "Path to a YAML or JSON config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--goose_tools_root",
        type=str,
        default=DEFAULT_GOOSE_TOOLS_ROOT,
        help="Directory that contains the goosetools package.",
    )
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument("--run_name", type=str, default="convnext_mask2former")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gradient_checkpointing",
                        action="store_true",
                        help=gradient_checkpointing_help)

    parser.add_argument("--resize_width", type=int, default=512)
    parser.add_argument("--resize_height", type=int, default=512)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument(
        "--convnext_model_name_or_path",
        type=str,
        default="facebook/dinov3-convnext-large-pretrain-lvd1689m",
        help=convnext_help,
    )
    parser.add_argument(
        "--mask2former_pretrained_model_name_or_path",
        type=str,
        default="facebook/mask2former-swin-large-ade-semantic",
        help=mask2former_help,
    )
    parser.add_argument("--freeze_encoder",
                        dest="freeze_encoder",
                        action="store_true",
                        help="Freeze the ConvNeXt encoder.")
    parser.add_argument(
        "--unfreeze_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Train the full ConvNeXt encoder with a lower lr.",
    )
    parser.add_argument("--freeze_mask2former_decoder",
                        action="store_true",
                        help=freeze_decoder_help)
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help=
        "Use focal classification loss and focal-aware Hungarian class cost.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha factor used by the Mask2Former focal classification loss.",
    )
    parser.add_argument(
        "--focal_class_alphas",
        type=float,
        nargs="+",
        default=None,
        help="Optional class-wise focal alpha values for semantic classes.",
    )
    parser.add_argument(
        "--focal_no_object_alpha",
        type=float,
        default=None,
        help="Optional focal alpha value for the no-object query class.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma factor used by the Mask2Former focal classification loss.",
    )
    parser.add_argument(
        "--focal_normalize_by_num_masks",
        dest="focal_normalize_by_num_masks",
        action="store_true",
        help="Normalize focal classification loss by the matched-mask count.",
    )
    parser.add_argument(
        "--disable_focal_num_masks_normalization",
        dest="focal_normalize_by_num_masks",
        action="store_false",
        help=
        "Average focal classification loss over queries instead of matched masks.",
    )
    parser.add_argument(
        "--feature_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help=
        ("ConvNeXt stage indices selected as Mask2Former pixel-decoder inputs. "
         "Include the highest-resolution stage, e.g. [0, 1, 2, 3], so the "
         "top-down FPN path can build high-resolution mask features."),
    )
    parser.add_argument(
        "--disable_encoder_norm",
        action="store_true",
        help="Disable ConvNeXt normalization from AutoImageProcessor stats.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint created by this script.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        help="Use DistributedSampler for validation during torchrun launches.",
    )
    parser.set_defaults(freeze_encoder=True)
    parser.set_defaults(focal_normalize_by_num_masks=True)
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config",
                               type=str,
                               default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args()

    parser = build_arg_parser()
    config_path = Path(config_args.config).expanduser().resolve()
    if config_path.exists():
        config_values = load_config_file(str(config_path))
        validate_config_keys(parser, config_values)
        parser.set_defaults(**config_values)

    args = parser.parse_args()
    args.config = str(Path(args.config).expanduser().resolve())

    if args.data_path is None:
        parser.error(
            "`data_path` is required. Set it in the config file or pass it on the command line."
        )

    return args

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from goose.runtime import EXPERIMENT_ROOT, ensure_project_paths

ensure_project_paths()

from args import build_arg_parser as build_base_arg_parser  # noqa: E402
from args import load_config_file, validate_config_keys  # noqa: E402

DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "config" / "train.yaml"


def default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(EXPERIMENT_ROOT / "outputs" / timestamp)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser()
    parser.description = "Experimental DINOv3 Adapter + upstream Mask2Former trainer"
    parser.set_defaults(run_name="dinov3_seg_upstream")
    parser.set_defaults(output_dir=default_output_dir())
    parser.set_defaults(config=str(DEFAULT_CONFIG_PATH))

    parser.add_argument(
        "--dinov3_loader",
        type=str,
        choices=["torch_hub", "hf_auto"],
        default="torch_hub",
        help=(
            "How to load the DINOv3 ViT backbone. "
            "`torch_hub` matches the upstream facebookresearch/dinov3 usage; "
            "`hf_auto` is a fallback for compatible HF checkpoints."
        ),
    )
    parser.add_argument(
        "--dinov3_repo_or_dir",
        type=str,
        default="facebookresearch/dinov3",
        help=(
            "torch.hub repo or local clone for DINOv3. "
            "Use a filesystem path together with --dinov3_source local when "
            "you have a checked-out DINOv3 repository."
        ),
    )
    parser.add_argument(
        "--dinov3_source",
        type=str,
        choices=["github", "local"],
        default="github",
        help="torch.hub source mode used when --dinov3_loader torch_hub.",
    )
    parser.add_argument(
        "--dinov3_model_name",
        type=str,
        default="dinov3_vitl16",
        help="torch.hub DINOv3 entrypoint name.",
    )
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        default=None,
        help="Optional weights argument forwarded to the torch.hub DINOv3 constructor.",
    )
    parser.add_argument(
        "--dinov3_model_name_or_path",
        type=str,
        default=None,
        help="HF model id or local path used when --dinov3_loader hf_auto.",
    )
    parser.add_argument(
        "--dinov3_image_processor_name_or_path",
        type=str,
        default=None,
        help="Optional HF image processor reference used only to fetch mean/std.",
    )
    parser.add_argument(
        "--vit_feature_indices",
        type=int,
        nargs="+",
        default=[4, 11, 17, 23],
        help=(
            "ViT block indices used as the DINOv3 adapter interaction layers. "
            "For DINOv3 ViT-L/16 the upstream segmentation code uses [4, 11, 17, 23]."
        ),
    )
    parser.add_argument(
        "--vit_patch_size",
        type=int,
        default=16,
        help="Fallback ViT patch size used when the loaded backbone does not expose one.",
    )
    parser.add_argument(
        "--vit_norm_features",
        dest="vit_norm_features",
        action="store_true",
        help="Use norm=True when calling get_intermediate_layers on the DINOv3 backbone.",
    )
    parser.add_argument(
        "--disable_vit_feature_norm",
        dest="vit_norm_features",
        action="store_false",
        help="Disable norm=True when extracting DINOv3 intermediate features.",
    )
    parser.set_defaults(vit_norm_features=True)

    parser.add_argument(
        "--m2f_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension used by the vendored upstream Mask2Former head.",
    )
    parser.add_argument(
        "--m2f_no_object_weight",
        type=float,
        default=0.1,
        help="Weight applied to the no-object query class.",
    )
    parser.add_argument(
        "--m2f_class_weight",
        type=float,
        default=2.0,
        help="Classification loss and matching cost weight.",
    )
    parser.add_argument(
        "--m2f_mask_weight",
        type=float,
        default=5.0,
        help="Mask loss and matching cost weight.",
    )
    parser.add_argument(
        "--m2f_dice_weight",
        type=float,
        default=5.0,
        help="Dice loss and matching cost weight.",
    )
    parser.add_argument(
        "--m2f_train_num_points",
        type=int,
        default=12_544,
        help="Number of sampled points used by the upstream Mask2Former criterion.",
    )
    parser.add_argument(
        "--m2f_oversample_ratio",
        type=float,
        default=3.0,
        help="Oversample ratio used by the upstream Mask2Former criterion.",
    )
    parser.add_argument(
        "--m2f_importance_sample_ratio",
        type=float,
        default=0.75,
        help="Importance-sample ratio used by the upstream Mask2Former criterion.",
    )

    parser.add_argument(
        "--adapter_pretrain_size",
        type=int,
        default=512,
        help="Reference image size used by the vendored DINOv3 adapter positional interpolation.",
    )
    parser.add_argument(
        "--adapter_conv_inplane",
        type=int,
        default=64,
        help="Spatial prior module channel width used by the adapter.",
    )
    parser.add_argument(
        "--adapter_deform_num_heads",
        type=int,
        default=16,
        help="Number of deformable-attention heads used by the adapter.",
    )
    parser.add_argument(
        "--adapter_drop_path_rate",
        type=float,
        default=0.3,
        help="Drop-path rate used by the vendored adapter interaction blocks.",
    )
    parser.add_argument(
        "--adapter_init_values",
        type=float,
        default=0.0,
        help="Init-values argument forwarded to the vendored adapter blocks.",
    )
    parser.add_argument(
        "--adapter_with_cffn",
        dest="adapter_with_cffn",
        action="store_true",
        help="Enable the adapter convolutional FFN blocks.",
    )
    parser.add_argument(
        "--disable_adapter_cffn",
        dest="adapter_with_cffn",
        action="store_false",
        help="Disable the adapter convolutional FFN blocks.",
    )
    parser.set_defaults(adapter_with_cffn=True)
    parser.add_argument(
        "--adapter_cffn_ratio",
        type=float,
        default=0.25,
        help="Hidden-size ratio used by the adapter convolutional FFN blocks.",
    )
    parser.add_argument(
        "--adapter_deform_ratio",
        type=float,
        default=0.5,
        help="Deformable attention ratio used by the adapter blocks.",
    )
    parser.add_argument(
        "--adapter_add_vit_feature",
        dest="adapter_add_vit_feature",
        action="store_true",
        help="Fuse resized ViT features back into the adapter output pyramid.",
    )
    parser.add_argument(
        "--disable_adapter_vit_feature_fusion",
        dest="adapter_add_vit_feature",
        action="store_false",
        help="Disable final ViT feature fusion in the adapter output pyramid.",
    )
    parser.set_defaults(adapter_add_vit_feature=True)
    parser.add_argument(
        "--adapter_use_extra_extractor",
        dest="adapter_use_extra_extractor",
        action="store_true",
        help="Enable the extra extractor blocks in the last adapter stage.",
    )
    parser.add_argument(
        "--disable_adapter_extra_extractor",
        dest="adapter_use_extra_extractor",
        action="store_false",
        help="Disable the extra extractor blocks in the last adapter stage.",
    )
    parser.set_defaults(adapter_use_extra_extractor=True)

    parser.add_argument(
        "--model_type",
        type=str,
        default="experimental_dinov3_seg_upstream",
        help="Checkpoint metadata identifier for this experimental model family.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
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
    if len(args.vit_feature_indices) < 4:
        parser.error(
            "`vit_feature_indices` must contain at least 4 interaction layers for the upstream adapter."
        )
    if args.dinov3_loader == "hf_auto" and not args.dinov3_model_name_or_path:
        parser.error(
            "`--dinov3_model_name_or_path` is required when --dinov3_loader hf_auto."
        )
    return args

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from goose.runtime import ensure_project_paths

ensure_project_paths()

from utils import is_distributed_enabled, is_main_process  # noqa: E402
from vendor.mask2former_criterion import Mask2FormerSetCriterion  # noqa: E402
from vendor.models.backbone import DINOv3_Adapter  # noqa: E402
from vendor.models.heads import Mask2FormerHead  # noqa: E402


@dataclass
class DINOv3SegmentationOutput:
    loss: Optional[torch.Tensor]
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor
    auxiliary_logits: Optional[list[dict[str, torch.Tensor]]] = None
    loss_dict: Optional[dict[str, torch.Tensor]] = None


def load_matching_state_dict(
    module: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    module_name: str,
) -> None:
    target_state_dict = module.state_dict()
    matched_state_dict: Dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for name, value in source_state_dict.items():
        target_value = target_state_dict.get(name)
        if target_value is None:
            skipped_keys.append(name)
            continue
        if target_value.shape != value.shape:
            skipped_keys.append(
                f"{name} (source={tuple(value.shape)}, target={tuple(target_value.shape)})"
            )
            continue
        matched_state_dict[name] = value

    missing_keys = sorted(set(target_state_dict) - set(matched_state_dict))
    module.load_state_dict(matched_state_dict, strict=False)

    print(
        f"Loaded {len(matched_state_dict)}/{len(target_state_dict)} "
        f"{module_name} tensors from the pretrained Mask2Former checkpoint."
    )
    if skipped_keys:
        preview = ", ".join(skipped_keys[:5])
        suffix = " ..." if len(skipped_keys) > 5 else ""
        print(
            f"Skipped {module_name} tensors due to shape/key mismatch: "
            f"{preview}{suffix}"
        )
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        suffix = " ..." if len(missing_keys) > 5 else ""
        print(f"Newly initialized {module_name} tensors: {preview}{suffix}")


def _remap_conv_norm_sequence_key(
    key: str,
    *,
    source_prefix: str,
    target_prefix: str,
) -> Optional[str]:
    if not key.startswith(source_prefix):
        return None

    remainder = key[len(source_prefix) :]
    parts = remainder.split(".")
    if len(parts) < 3:
        return None

    index, submodule = parts[0], parts[1]
    field = ".".join(parts[2:])
    if submodule == "0":
        return f"{target_prefix}{index}.{field}"
    if submodule == "1":
        return f"{target_prefix}{index}.norm.{field}"
    return None


def remap_local_pixel_decoder_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}

    for name, value in source_state_dict.items():
        target_name: Optional[str] = None

        if name.startswith("input_projections."):
            target_name = name.replace("input_projections.", "input_convs.", 1)
        elif name.startswith("encoder.layers."):
            target_name = name.replace("encoder.layers.", "encoder.encoder.layers.", 1)
            target_name = target_name.replace(".self_attn_layer_norm.", ".norm1.")
            target_name = target_name.replace(".fc1.", ".linear1.")
            target_name = target_name.replace(".fc2.", ".linear2.")
            target_name = target_name.replace(".final_layer_norm.", ".norm2.")
        elif name == "level_embed":
            target_name = "encoder.level_encoding"
        elif name.startswith("mask_projection."):
            target_name = name.replace("mask_projection.", "mask_feature.", 1)
        elif name.startswith("lateral_convolutions."):
            target_name = _remap_conv_norm_sequence_key(
                name,
                source_prefix="lateral_convolutions.",
                target_prefix="lateral_convs.",
            )
        elif name.startswith("output_convolutions."):
            target_name = _remap_conv_norm_sequence_key(
                name,
                source_prefix="output_convolutions.",
                target_prefix="output_convs.",
            )

        if target_name is not None:
            remapped[target_name] = value

    return remapped


def remap_local_transformer_state_dict(
    transformer_state_dict: Dict[str, torch.Tensor],
    class_predictor_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}

    direct_mappings = {
        "queries_features.weight": "query_feat.weight",
        "queries_embedder.weight": "query_embed.weight",
        "level_embed.weight": "level_embed.weight",
        "decoder.layernorm.weight": "post_norm.weight",
        "decoder.layernorm.bias": "post_norm.bias",
    }
    for source_name, target_name in direct_mappings.items():
        value = transformer_state_dict.get(source_name)
        if value is not None:
            remapped[target_name] = value

    for source_name, target_name in {
        "weight": "class_embed.weight",
        "bias": "class_embed.bias",
    }.items():
        value = class_predictor_state_dict.get(source_name)
        if value is not None:
            remapped[target_name] = value

    for index in range(3):
        for field in ("weight", "bias"):
            source_name = f"decoder.mask_predictor.mask_embedder.{index}.0.{field}"
            target_name = f"mask_embed.layers.{index}.{field}"
            value = transformer_state_dict.get(source_name)
            if value is not None:
                remapped[target_name] = value

    for index in range(3):
        for field in ("weight", "bias"):
            source_name = f"input_projections.{index}.{field}"
            target_name = f"input_proj.{index}.{field}"
            value = transformer_state_dict.get(source_name)
            if value is not None:
                remapped[target_name] = value

    decoder_layer_prefix = "decoder.layers."
    layer_indices = sorted(
        {
            int(key[len(decoder_layer_prefix) :].split(".", 1)[0])
            for key in transformer_state_dict
            if key.startswith(decoder_layer_prefix)
        }
    )

    for layer_index in layer_indices:
        source_prefix = f"decoder.layers.{layer_index}"

        q_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.q_proj.weight"
        )
        k_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.k_proj.weight"
        )
        v_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.v_proj.weight"
        )
        if (
            q_proj_weight is not None
            and k_proj_weight is not None
            and v_proj_weight is not None
        ):
            remapped[
                f"transformer_self_attention_layers.{layer_index}.self_attn.in_proj_weight"
            ] = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)

        q_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.q_proj.bias"
        )
        k_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.k_proj.bias"
        )
        v_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.v_proj.bias"
        )
        if (
            q_proj_bias is not None
            and k_proj_bias is not None
            and v_proj_bias is not None
        ):
            remapped[
                f"transformer_self_attention_layers.{layer_index}.self_attn.in_proj_bias"
            ] = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0)

        for source_name, target_name in {
            "self_attn.out_proj.weight": "transformer_self_attention_layers.{i}.self_attn.out_proj.weight",
            "self_attn.out_proj.bias": "transformer_self_attention_layers.{i}.self_attn.out_proj.bias",
            "self_attn_layer_norm.weight": "transformer_self_attention_layers.{i}.norm.weight",
            "self_attn_layer_norm.bias": "transformer_self_attention_layers.{i}.norm.bias",
            "cross_attn.in_proj_weight": "transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight",
            "cross_attn.in_proj_bias": "transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias",
            "cross_attn.out_proj.weight": "transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight",
            "cross_attn.out_proj.bias": "transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias",
            "cross_attn_layer_norm.weight": "transformer_cross_attention_layers.{i}.norm.weight",
            "cross_attn_layer_norm.bias": "transformer_cross_attention_layers.{i}.norm.bias",
            "fc1.weight": "transformer_ffn_layers.{i}.linear1.weight",
            "fc1.bias": "transformer_ffn_layers.{i}.linear1.bias",
            "fc2.weight": "transformer_ffn_layers.{i}.linear2.weight",
            "fc2.bias": "transformer_ffn_layers.{i}.linear2.bias",
            "final_layer_norm.weight": "transformer_ffn_layers.{i}.norm.weight",
            "final_layer_norm.bias": "transformer_ffn_layers.{i}.norm.bias",
        }.items():
            value = transformer_state_dict.get(f"{source_prefix}.{source_name}")
            if value is not None:
                remapped[target_name.format(i=layer_index)] = value

    return remapped


def _lookup_conv_in_channels(
    state_dict: Dict[str, torch.Tensor],
    candidate_keys: Sequence[str],
) -> int:
    for key in candidate_keys:
        value = state_dict.get(key)
        if value is not None and value.ndim >= 2:
            return int(value.shape[1])
    raise KeyError(f"Could not find any of the expected keys: {candidate_keys}")


def infer_pretrained_feature_channels(
    pretrained_mask2former,
) -> Optional[Dict[str, int]]:
    try:
        decoder_state_dict = pretrained_mask2former.model.pixel_level_module.decoder.state_dict()
        return {
            "1": _lookup_conv_in_channels(
                decoder_state_dict,
                ("lateral_convolutions.0.0.weight", "lateral_convolutions.0.weight"),
            ),
            "2": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.2.0.weight", "input_projections.2.weight"),
            ),
            "3": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.1.0.weight", "input_projections.1.weight"),
            ),
            "4": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.0.0.weight", "input_projections.0.weight"),
            ),
        }
    except (AttributeError, KeyError):
        return None


def build_feature_channel_aligner(
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()

    aligner = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )
    with torch.no_grad():
        aligner.weight.zero_()
        aligner.bias.zero_()
        for out_index in range(out_channels):
            in_index = min((out_index * in_channels) // out_channels, in_channels - 1)
            aligner.weight[out_index, in_index, 0, 0] = 1.0
    return aligner


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value)


def infer_embed_dim(model: nn.Module) -> int:
    candidates = [
        model,
        getattr(model, "backbone", None),
        getattr(model, "config", None),
        getattr(getattr(model, "backbone", None), "config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("embed_dim", "hidden_size", "num_features", "hidden_dim"):
            value = getattr(candidate, attribute, None)
            if value is not None:
                return int(value)

    pos_embed = getattr(model, "pos_embed", None)
    if isinstance(pos_embed, torch.Tensor):
        return int(pos_embed.shape[-1])

    raise ValueError(
        "Could not infer the DINOv3 embedding dimension from the loaded backbone."
    )


def infer_patch_size(model: nn.Module, fallback: int) -> int:
    candidates = [
        model,
        getattr(model, "backbone", None),
        getattr(model, "config", None),
        getattr(getattr(model, "backbone", None), "config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("patch_size", "patch_embed_patch_size"):
            value = _as_int(getattr(candidate, attribute, None))
            if value is not None:
                return value
        patch_embed = getattr(candidate, "patch_embed", None)
        if patch_embed is not None:
            value = _as_int(getattr(patch_embed, "patch_size", None))
            if value is not None:
                return value
            projection = getattr(patch_embed, "proj", None)
            if projection is not None:
                value = _as_int(getattr(projection, "kernel_size", None))
                if value is not None:
                    return value
    return int(fallback)


class DINOInputNormalizer(nn.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float], enabled: bool):
        super().__init__()
        self.enabled = enabled
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.max().detach().item() > 1.5:
            x = x / 255.0
        return (x - self.mean) / self.std


class HFBackboneIntermediateLayersAdapter(nn.Module):
    def __init__(self, model: nn.Module, embed_dim: int, patch_size: int):
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.num_hidden_layers = int(getattr(self.config, "num_hidden_layers", 0))
        self.num_register_tokens = int(
            getattr(
                self.config,
                "num_register_tokens",
                getattr(self.config, "n_storage_tokens", 0) or 0,
            )
            or 0
        )
        self.untie_cls_and_patch_norms = bool(
            getattr(model, "untie_cls_and_patch_norms", False)
        )
        self.norm = getattr(model, "norm", None)
        self.cls_norm = getattr(model, "cls_norm", None)

    def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.untie_cls_and_patch_norms and self.cls_norm is not None and self.norm is not None:
            cls_and_registers = self.cls_norm(
                hidden_state[:, : self.num_register_tokens + 1]
            )
            patch_tokens = self.norm(hidden_state[:, self.num_register_tokens + 1 :])
            return torch.cat((cls_and_registers, patch_tokens), dim=1)
        if self.norm is not None:
            return self.norm(hidden_state)
        return hidden_state

    def _resolve_layer_indexes(self, n: Union[int, Sequence[int]]) -> list[int]:
        if isinstance(n, int):
            if self.num_hidden_layers <= 0:
                raise ValueError(
                    "Could not infer the number of hidden layers for the Hugging Face DINOv3 backbone."
                )
            if n <= 0 or n > self.num_hidden_layers:
                raise ValueError(
                    f"Requested {n} intermediate layers, but the backbone only has {self.num_hidden_layers} blocks."
                )
            return list(range(self.num_hidden_layers - n, self.num_hidden_layers))

        layer_indexes = [int(index) for index in n]
        if not layer_indexes:
            raise ValueError("At least one intermediate layer index must be provided.")

        for index in layer_indexes:
            if index < 0 or (self.num_hidden_layers > 0 and index >= self.num_hidden_layers):
                raise ValueError(
                    f"Intermediate layer index {index} is out of range for a backbone with "
                    f"{self.num_hidden_layers} blocks."
                )
        return layer_indexes

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ):
        layer_indexes = self._resolve_layer_indexes(n)
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "The Hugging Face DINOv3 backbone did not return hidden states. "
                "This experimental adapter requires `output_hidden_states=True` support."
            )

        selected_hidden_states = [hidden_states[index + 1] for index in layer_indexes]
        if norm:
            selected_hidden_states = [
                self._normalize_hidden_state(hidden_state)
                for hidden_state in selected_hidden_states
            ]

        class_tokens = [hidden_state[:, 0] for hidden_state in selected_hidden_states]
        extra_tokens = [
            hidden_state[:, 1 : self.num_register_tokens + 1]
            for hidden_state in selected_hidden_states
        ]
        patch_tokens = [
            hidden_state[:, self.num_register_tokens + 1 :]
            for hidden_state in selected_hidden_states
        ]

        if reshape:
            batch_size, _, height, width = x.shape
            patch_tokens = [
                patch_token.reshape(
                    batch_size,
                    height // self.patch_size,
                    width // self.patch_size,
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                for patch_token in patch_tokens
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_tokens)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(patch_tokens, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(patch_tokens, extra_tokens))
        return tuple(zip(patch_tokens, class_tokens, extra_tokens))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class UpstreamDINOv3SegmentationModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
            from transformers.models.mask2former.configuration_mask2former import (
                Mask2FormerConfig,
            )
            from models.mask2former_model import (
                Mask2FormerForUniversalSegmentation,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with `pip install transformers>=4.56.0`."
            ) from exc

        self._auto_image_processor_class = AutoImageProcessor
        self._auto_model_class = AutoModel
        self._auto_config_class = AutoConfig
        self._mask2former_config_class = Mask2FormerConfig
        self._mask2former_model_class = Mask2FormerForUniversalSegmentation

        local_files_only = is_distributed_enabled() and not is_main_process()
        encoder, encoder_hidden_size, patch_size, processor_name = self._load_dinov3_encoder(
            args, local_files_only
        )
        pretrained_mask2former = self._load_pretrained_mask2former(
            args=args,
            id2label=id2label,
            label2id=label2id,
            local_files_only=local_files_only,
        )
        pretrained_mask2former_config = (
            pretrained_mask2former.config if pretrained_mask2former is not None else None
        )
        pretrained_feature_channels = (
            infer_pretrained_feature_channels(pretrained_mask2former)
            if pretrained_mask2former is not None
            else None
        )

        image_mean, image_std = self._load_image_stats(processor_name, local_files_only)
        self.normalizer = DINOInputNormalizer(
            mean=image_mean,
            std=image_std,
            enabled=not getattr(args, "disable_encoder_norm", False),
        )

        self.segmentation_backbone = DINOv3_Adapter(
            encoder,
            interaction_indexes=list(args.vit_feature_indices),
            pretrain_size=int(getattr(args, "adapter_pretrain_size", max(args.resize_height, args.resize_width))),
            conv_inplane=int(getattr(args, "adapter_conv_inplane", 64)),
            deform_num_heads=int(getattr(args, "adapter_deform_num_heads", 16)),
            drop_path_rate=float(getattr(args, "adapter_drop_path_rate", 0.3)),
            init_values=float(getattr(args, "adapter_init_values", 0.0)),
            with_cffn=bool(getattr(args, "adapter_with_cffn", True)),
            cffn_ratio=float(getattr(args, "adapter_cffn_ratio", 0.25)),
            deform_ratio=float(getattr(args, "adapter_deform_ratio", 0.5)),
            add_vit_feature=bool(getattr(args, "adapter_add_vit_feature", True)),
            use_extra_extractor=bool(getattr(args, "adapter_use_extra_extractor", True)),
            with_cp=bool(getattr(args, "gradient_checkpointing", False)),
            freeze_backbone=bool(getattr(args, "freeze_encoder", True)),
        )

        requested_hidden_dim = int(getattr(args, "m2f_hidden_dim", 256))
        if pretrained_mask2former_config is not None:
            pretrained_hidden_dim = int(pretrained_mask2former_config.hidden_dim)
            if requested_hidden_dim != pretrained_hidden_dim:
                print(
                    "[INFO] Overriding m2f_hidden_dim={} with pretrained "
                    "Mask2Former hidden_dim={} for compatibility.".format(
                        requested_hidden_dim,
                        pretrained_hidden_dim,
                    )
                )
            head_hidden_dim = pretrained_hidden_dim
            mask_dim = int(
                getattr(
                    pretrained_mask2former_config,
                    "mask_feature_size",
                    pretrained_hidden_dim,
                )
            )
            transformer_dropout = float(
                getattr(pretrained_mask2former_config, "dropout", 0.0)
            )
            transformer_nheads = int(
                getattr(pretrained_mask2former_config, "num_attention_heads", 8)
            )
            transformer_dim_feedforward = int(
                getattr(pretrained_mask2former_config, "encoder_feedforward_dim", 1024)
            )
            transformer_enc_layers = int(
                getattr(pretrained_mask2former_config, "encoder_layers", 6)
            )
            predictor_nheads = int(
                getattr(pretrained_mask2former_config, "num_attention_heads", 8)
            )
            predictor_dim_feedforward = int(
                getattr(pretrained_mask2former_config, "dim_feedforward", 2048)
            )
            predictor_dec_layers = max(
                int(getattr(pretrained_mask2former_config, "decoder_layers", 10)) - 1,
                1,
            )
            predictor_pre_norm = bool(
                getattr(pretrained_mask2former_config, "pre_norm", False)
            )
            predictor_enforce_input_project = bool(
                getattr(
                    pretrained_mask2former_config,
                    "enforce_input_projection",
                    False,
                )
            )
            common_stride = int(
                getattr(pretrained_mask2former_config, "common_stride", 4)
            )
        else:
            head_hidden_dim = requested_hidden_dim
            mask_dim = head_hidden_dim
            transformer_dropout = 0.0
            transformer_nheads = 16
            transformer_dim_feedforward = 4096
            transformer_enc_layers = 6
            predictor_nheads = 16
            predictor_dim_feedforward = 4096
            predictor_dec_layers = 9
            predictor_pre_norm = False
            predictor_enforce_input_project = False
            common_stride = 4

        if pretrained_feature_channels is not None:
            head_input_channels = pretrained_feature_channels
            if len(set(head_input_channels.values())) > 1:
                print(
                    "[INFO] Aligning DINO adapter feature channels to the "
                    "pretrained Mask2Former pyramid: "
                    f"{head_input_channels}"
                )
        else:
            head_input_channels = {
                "1": encoder_hidden_size,
                "2": encoder_hidden_size,
                "3": encoder_hidden_size,
                "4": encoder_hidden_size,
            }

        self.head_feature_aligners = nn.ModuleDict(
            {
                level: build_feature_channel_aligner(
                    encoder_hidden_size,
                    head_input_channels[level],
                )
                for level in ("1", "2", "3", "4")
            }
        )

        self.segmentation_head = Mask2FormerHead(
            input_shape={
                "1": [head_input_channels["1"], patch_size * 4, patch_size * 4, 4],
                "2": [head_input_channels["2"], patch_size * 2, patch_size * 2, 8],
                "3": [head_input_channels["3"], patch_size, patch_size, 16],
                "4": [head_input_channels["4"], int(patch_size / 2), int(patch_size / 2), 32],
            },
            hidden_dim=head_hidden_dim,
            num_classes=int(args.num_classes),
            ignore_value=int(args.ignore_index),
            mask_dim=mask_dim,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            predictor_nheads=predictor_nheads,
            predictor_dim_feedforward=predictor_dim_feedforward,
            predictor_dec_layers=predictor_dec_layers,
            predictor_pre_norm=predictor_pre_norm,
            predictor_enforce_input_project=predictor_enforce_input_project,
            common_stride=common_stride,
        )
        if pretrained_mask2former is not None:
            self._initialize_segmentation_head_from_pretrained(pretrained_mask2former)

        if getattr(args, "freeze_mask2former_decoder", False):
            for parameter in self.segmentation_head.parameters():
                parameter.requires_grad = False

        self.criterion = Mask2FormerSetCriterion(
            num_classes=int(args.num_classes),
            no_object_weight=float(getattr(args, "m2f_no_object_weight", 0.1)),
            class_weight=float(getattr(args, "m2f_class_weight", 2.0)),
            mask_weight=float(getattr(args, "m2f_mask_weight", 5.0)),
            dice_weight=float(getattr(args, "m2f_dice_weight", 5.0)),
            num_points=int(getattr(args, "m2f_train_num_points", 12_544)),
            oversample_ratio=float(getattr(args, "m2f_oversample_ratio", 3.0)),
            importance_sample_ratio=float(
                getattr(args, "m2f_importance_sample_ratio", 0.75)
            ),
            ignore_index=int(args.ignore_index),
            use_focal_loss=bool(getattr(args, "use_focal_loss", False)),
            focal_alpha=float(getattr(args, "focal_alpha", 0.25)),
            focal_class_alphas=getattr(args, "focal_class_alphas", None),
            focal_no_object_alpha=getattr(args, "focal_no_object_alpha", None),
            focal_gamma=float(getattr(args, "focal_gamma", 2.0)),
            focal_normalize_by_num_masks=bool(
                getattr(args, "focal_normalize_by_num_masks", True)
            ),
            classification_loss_type=getattr(args, "classification_loss_type", None),
            seesaw_p=float(getattr(args, "seesaw_p", 0.8)),
            seesaw_q=float(getattr(args, "seesaw_q", 2.0)),
            seesaw_eps=float(getattr(args, "seesaw_eps", 1e-2)),
        )

        self.patch_size = patch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.inference_size = (int(args.resize_height), int(args.resize_width))

    def _load_pretrained_mask2former(
        self,
        args: argparse.Namespace,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        local_files_only: bool,
    ):
        del id2label, label2id

        pretrained_name_or_path = getattr(
            args,
            "mask2former_pretrained_model_name_or_path",
            None,
        )
        if not pretrained_name_or_path:
            return None

        try:
            config = self._mask2former_config_class.from_pretrained(
                pretrained_name_or_path,
                local_files_only=local_files_only,
            )
            model = self._mask2former_model_class.from_pretrained(
                pretrained_name_or_path,
                config=config,
                local_files_only=local_files_only,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the pretrained Mask2Former checkpoint from "
                f"{pretrained_name_or_path!r}. Set "
                "`mask2former_pretrained_model_name_or_path` to a valid local path "
                "or cached Hugging Face model before training."
            ) from exc

        model.eval()
        return model

    def _initialize_segmentation_head_from_pretrained(
        self,
        pretrained_mask2former,
    ) -> None:
        pixel_decoder_state_dict = remap_local_pixel_decoder_state_dict(
            pretrained_mask2former.model.pixel_level_module.decoder.state_dict()
        )
        load_matching_state_dict(
            self.segmentation_head.pixel_decoder,
            pixel_decoder_state_dict,
            "experimental pixel decoder",
        )

        transformer_state_dict = remap_local_transformer_state_dict(
            pretrained_mask2former.model.transformer_module.state_dict(),
            pretrained_mask2former.class_predictor.state_dict(),
        )
        load_matching_state_dict(
            self.segmentation_head.predictor,
            transformer_state_dict,
            "experimental transformer decoder",
        )

    def _load_dinov3_encoder(
        self,
        args: argparse.Namespace,
        local_files_only: bool,
    ) -> Tuple[nn.Module, int, int, Optional[str]]:
        def infer_hf_model_id() -> Optional[str]:
            explicit = getattr(args, "dinov3_model_name_or_path", None)
            if explicit:
                return explicit

            hub_name = str(getattr(args, "dinov3_model_name", "")).strip().lower()
            weights_name = str(getattr(args, "dinov3_weights", "") or "").strip().lower()
            if weights_name and weights_name not in {"lvd1689m", "sat493m"}:
                return None

            mapping = {
                "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "dinov3_vitl16plus": "facebook/dinov3-vitl16plus-pretrain-lvd1689m",
                "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            }
            return mapping.get(hub_name)

        def load_via_hf(model_id: str) -> Tuple[nn.Module, int, int, Optional[str]]:
            encoder_config = self._auto_config_class.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            encoder = self._auto_model_class.from_pretrained(
                model_id,
                config=encoder_config,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            processor_name = args.dinov3_image_processor_name_or_path or model_id
            encoder_hidden_size = infer_embed_dim(encoder)
            patch_size = infer_patch_size(encoder, args.vit_patch_size)
            return encoder, encoder_hidden_size, patch_size, processor_name

        if args.dinov3_loader == "hf_auto":
            model_id = infer_hf_model_id()
            if model_id is None:
                raise ValueError(
                    "Could not infer a Hugging Face DINOv3 ViT model id. "
                    "Please set --dinov3_model_name_or_path explicitly."
                )
            encoder, encoder_hidden_size, patch_size, processor_name = load_via_hf(
                model_id
            )
        else:
            hub_kwargs = {}
            if args.dinov3_weights is not None:
                hub_kwargs["weights"] = args.dinov3_weights
            try:
                encoder = torch.hub.load(
                    args.dinov3_repo_or_dir,
                    args.dinov3_model_name,
                    source=args.dinov3_source,
                    **hub_kwargs,
                )
            except TypeError as exc:
                if args.dinov3_weights is not None:
                    raise TypeError(
                        "The selected DINOv3 torch.hub entrypoint did not accept "
                        "a `weights=` argument. Try removing `--dinov3_weights` "
                        "or changing `--dinov3_model_name`."
                    ) from exc
                raise
            except Exception as exc:
                fallback_model_id = infer_hf_model_id()
                if fallback_model_id is None:
                    raise

                syntax_related = isinstance(exc, (SyntaxError, TypeError))
                running_py39 = sys.version_info < (3, 10)
                if not (running_py39 or syntax_related):
                    raise

                print(
                    "Warning: torch.hub loading of the local DINOv3 repo failed "
                    f"under Python {sys.version_info.major}.{sys.version_info.minor}. "
                    f"Falling back to Hugging Face model {fallback_model_id}. "
                    f"Original error: {exc}"
                )
                encoder, encoder_hidden_size, patch_size, processor_name = load_via_hf(
                    fallback_model_id
                )
            else:
                processor_name = args.dinov3_image_processor_name_or_path
                encoder_hidden_size = infer_embed_dim(encoder)
                patch_size = infer_patch_size(encoder, args.vit_patch_size)

        if not hasattr(encoder, "get_intermediate_layers"):
            encoder = HFBackboneIntermediateLayersAdapter(
                encoder,
                embed_dim=encoder_hidden_size,
                patch_size=patch_size,
            )

        return encoder, encoder_hidden_size, patch_size, processor_name

    def _load_image_stats(
        self,
        processor_name: Optional[str],
        local_files_only: bool,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        if processor_name is None:
            print(
                "Warning: no DINOv3 image processor reference was provided. "
                "Falling back to ImageNet normalization stats."
            )
            return image_mean, image_std

        try:
            processor = self._auto_image_processor_class.from_pretrained(
                processor_name,
                local_files_only=local_files_only,
            )
            image_mean = getattr(processor, "image_mean", image_mean)
            image_std = getattr(processor, "image_std", image_std)
        except Exception as exc:
            print(
                "Warning: failed to load image processor from "
                f"{processor_name}. Falling back to ImageNet normalization stats. "
                f"Original error: {exc}"
            )
        return image_mean, image_std

    def _build_targets(
        self,
        class_labels: Sequence[torch.Tensor],
        mask_labels: Sequence[torch.Tensor],
    ) -> List[dict[str, torch.Tensor]]:
        return [
            {
                "labels": labels.to(torch.int64),
                "masks": masks.to(torch.float32),
            }
            for labels, masks in zip(class_labels, mask_labels)
        ]

    def _compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        class_labels: Sequence[torch.Tensor],
        mask_labels: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        targets = self._build_targets(class_labels, mask_labels)
        loss_dict = self.criterion.compute_loss_dict(
            self.criterion._cast_outputs_to_float(predictions),
            targets,
        )

        total_loss = predictions["pred_logits"].sum() * 0.0
        for key, value in loss_dict.items():
            if key.startswith("loss_ce"):
                total_loss = total_loss + value * self.criterion.class_weight
            elif key.startswith("loss_mask"):
                total_loss = total_loss + value * self.criterion.mask_weight
            elif key.startswith("loss_dice"):
                total_loss = total_loss + value * self.criterion.dice_weight
        return total_loss, loss_dict

    def forward_features(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.normalizer(pixel_values)
        features = self.segmentation_backbone(normalized)
        return {
            level: self.head_feature_aligners[level](features[level])
            for level in ("1", "2", "3", "4")
        }

    def predict(
        self,
        pixel_values: torch.Tensor,
        rescale_to: Optional[Tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        features = self.forward_features(pixel_values)
        if rescale_to is None:
            rescale_to = tuple(int(x) for x in pixel_values.shape[-2:])
        return self.segmentation_head.predict(features, rescale_to=rescale_to)

    def forward(
        self,
        pixel_values: torch.Tensor,
        class_labels: Optional[Sequence[torch.Tensor]] = None,
        mask_labels: Optional[Sequence[torch.Tensor]] = None,
        **kwargs,
    ) -> DINOv3SegmentationOutput:
        del kwargs
        features = self.forward_features(pixel_values)
        predictions = self.segmentation_head(features)

        loss = None
        loss_dict = None
        if class_labels is not None and mask_labels is not None:
            loss, loss_dict = self._compute_loss(
                predictions,
                class_labels=class_labels,
                mask_labels=mask_labels,
            )

        return DINOv3SegmentationOutput(
            loss=loss,
            class_queries_logits=predictions["pred_logits"],
            masks_queries_logits=predictions["pred_masks"],
            auxiliary_logits=predictions.get("aux_outputs"),
            loss_dict=loss_dict,
        )

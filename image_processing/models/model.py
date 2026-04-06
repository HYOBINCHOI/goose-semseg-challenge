import argparse
import os
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn


def is_distributed_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# Reuse as much of the pretrained Mask2Former weights as possible while safely skipping mismatched layers
def load_matching_state_dict(module: nn.Module, source_state_dict,
                             module_name: str) -> None:
    target_state_dict = module.state_dict()
    matched_state_dict = {}
    skipped_keys = []

    for name, value in source_state_dict.items():
        target_value = target_state_dict.get(name)
        if target_value is None:
            skipped_keys.append(name)
            continue
        if target_value.shape != value.shape:
            skipped_keys.append(f"{name} (source={tuple(value.shape)}, "
                                f"target={tuple(target_value.shape)})")
            continue
        matched_state_dict[name] = value

    missing_keys = sorted(set(target_state_dict) - set(matched_state_dict))
    module.load_state_dict(matched_state_dict, strict=False)

    print(f"Loaded {len(matched_state_dict)}/{len(target_state_dict)} "
          f"{module_name} tensors from the pretrained Mask2Former checkpoint.")
    if skipped_keys:
        preview = ", ".join(skipped_keys[:5])
        suffix = " ..." if len(skipped_keys) > 5 else ""
        print(f"Skipped {module_name} tensors due to shape/key mismatch: "
              f"{preview}{suffix}")
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        suffix = " ..." if len(missing_keys) > 5 else ""
        print(f"Newly initialized {module_name} tensors: {preview}{suffix}")


class DINOInputNormalizer(nn.Module):

    def __init__(self, mean: Sequence[float], std: Sequence[float],
                 enabled: bool):
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


class ConvNeXtPixelLevelModuleBoosted(nn.Module):

    def __init__(
        self,
        config,
        encoder: nn.Module,
        normalizer: nn.Module,
        feature_indices: Sequence[int],
        encoder_hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.encoder = encoder
        self.normalizer = normalizer
        self.feature_indices = list(feature_indices)
        self.encoder_hidden_sizes = list(encoder_hidden_sizes)
        try:
            from .mask2former_model import Mask2FormerPixelDecoder
        except ImportError as exc:
            raise ImportError("transformers is required. Install with "
                              "`pip install transformers`.") from exc

        # The original Mask2Former pixel decoder uses the last 3 levels for
        # deformable attention and the remaining high-resolution level(s) for
        # the top-down FPN path. Without the extra stage, the semantic
        # information cannot be propagated to a finer resolution feature map.
        if len(self.feature_indices) < 4:
            raise ValueError(
                "Mask2Former's deformable-attention FPN path requires at "
                "least 4 feature levels. Set feature_indices to include the "
                "highest-resolution stage as well, e.g. [0, 1, 2, 3].")

        self.decoder = Mask2FormerPixelDecoder(
            config,
            feature_channels=self.encoder_hidden_sizes,
        )

    def _select_feature_maps(
            self, feature_maps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feature_maps) == len(self.feature_indices):
            for feature_map in feature_maps:
                if feature_map.ndim != 4:
                    raise ValueError(
                        "Expected each ConvNeXt feature map to be 4D, "
                        f"but got shape {tuple(feature_map.shape)}.")
            return list(feature_maps)

        selected = []
        for index in self.feature_indices:
            if index >= len(feature_maps) or index < -len(feature_maps):
                raise IndexError(f"feature index {index} is out of range for "
                                 f"{len(feature_maps)} ConvNeXt feature maps")
            feature_map = feature_maps[index]
            if feature_map.ndim != 4:
                raise ValueError(
                    f"Expected ConvNeXt feature map at index {index} "
                    f"to be 4D, but got shape {tuple(feature_map.shape)}.")
            selected.append(feature_map)
        return selected

    def _extract_feature_maps(self, encoder_outputs) -> Sequence[torch.Tensor]:
        feature_maps = getattr(encoder_outputs, "feature_maps", None)
        if feature_maps is not None:
            return feature_maps

        hidden_states = getattr(encoder_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("The selected ConvNeXt checkpoint did not return "
                             "feature maps or hidden states.")

        valid_stage_channels = set(self.encoder_hidden_sizes)
        spatial_hidden_states = [
            hidden_state for hidden_state in hidden_states
            if hidden_state.ndim == 4
            and hidden_state.shape[1] in valid_stage_channels
        ]
        if not spatial_hidden_states:
            raise ValueError(
                "The selected ConvNeXt checkpoint returned hidden states, "
                "but none matched the expected ConvNeXt stage channels.")

        deduplicated_feature_maps: List[torch.Tensor] = []
        for hidden_state in spatial_hidden_states:
            if (deduplicated_feature_maps
                    and deduplicated_feature_maps[-1].shape[-2:]
                    == hidden_state.shape[-2:]):
                deduplicated_feature_maps[-1] = hidden_state
            else:
                deduplicated_feature_maps.append(hidden_state)

        deduplicated_feature_maps = [
            hidden_state for hidden_state in deduplicated_feature_maps
            if hidden_state.shape[1] in valid_stage_channels
        ]
        return tuple(deduplicated_feature_maps)

    def forward(self,
                pixel_values: torch.Tensor,
                output_hidden_states: bool = False,
                **kwargs):
        try:
            from .mask2former_model import Mask2FormerPixelLevelModuleOutput
        except ImportError as exc:
            raise ImportError("transformers is required. Install with "
                              "`pip install transformers`.") from exc

        normalized_pixel_values = self.normalizer(pixel_values)

        encoder_outputs = self.encoder(
            pixel_values=normalized_pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_maps = self._extract_feature_maps(encoder_outputs)
        selected_feature_maps = self._select_feature_maps(feature_maps)
        decoder_output = self.decoder(
            selected_feature_maps,
            output_hidden_states=output_hidden_states,
        )

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=selected_feature_maps[-1],
            encoder_hidden_states=(tuple(selected_feature_maps)
                                   if output_hidden_states else None),
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class ConvNeXtMask2FormerBoostedModel(nn.Module):

    def __init__(
        self,
        args: argparse.Namespace,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ):
        super().__init__()
        try:
            from transformers import (AutoBackbone, AutoConfig,
                                      AutoImageProcessor, AutoModel)
            from transformers.models.mask2former.configuration_mask2former import \
                Mask2FormerConfig

            from .mask2former_model import Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise ImportError("transformers is required. Install with "
                              "`pip install transformers>=4.56.0`.") from exc

        local_files_only = is_distributed_enabled() and not is_main_process()
        encoder_config = AutoConfig.from_pretrained(
            args.convnext_model_name_or_path,
            local_files_only=local_files_only,
        )
        hidden_sizes = list(getattr(encoder_config, "hidden_sizes", []))
        if not hidden_sizes:
            raise ValueError("Could not infer ConvNeXt hidden_sizes "
                             "from the encoder config.")
        if len(args.feature_indices) < 4:
            raise ValueError(
                "feature_indices must include the highest-resolution "
                "backbone stage so Mask2Former can build its top-down FPN "
                "path. Use at least 4 stages, e.g. [0, 1, 2, 3].")

        encoder_hidden_sizes = []
        for index in args.feature_indices:
            if index >= len(hidden_sizes) or index < -len(hidden_sizes):
                raise IndexError(f"feature index {index} is out of range for "
                                 f"ConvNeXt hidden sizes {hidden_sizes}")
            encoder_hidden_sizes.append(hidden_sizes[index])

        out_indices = tuple(sorted(set(args.feature_indices)))

        if encoder_config.model_type == "dinov3_convnext":
            encoder = AutoModel.from_pretrained(
                args.convnext_model_name_or_path,
                local_files_only=local_files_only,
            )
        else:
            # Hugging Face ConvNeXt backbones expose stage_names including an
            # extra "stem" entry before the four ConvNeXt stages. Our
            # feature_indices refer to ConvNeXt stages [stage1..stage4], so
            # shift the requested indices by one when configuring AutoBackbone.
            stage_names = list(getattr(encoder_config, "stage_names", []))
            if len(stage_names
                   ) == len(hidden_sizes) + 1 and stage_names[0] == "stem":
                out_indices = tuple(
                    sorted(
                        set(index + 1 if index >= 0 else index
                            for index in args.feature_indices)))
            encoder = AutoBackbone.from_pretrained(
                args.convnext_model_name_or_path,
                out_indices=out_indices,
                local_files_only=local_files_only,
            )

        if args.freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        try:
            processor = AutoImageProcessor.from_pretrained(
                args.convnext_model_name_or_path,
                local_files_only=local_files_only,
            )
            image_mean = getattr(processor, "image_mean", image_mean)
            image_std = getattr(processor, "image_std", image_std)
        except Exception as exc:
            print("Warning: failed to load ConvNeXt image processor from "
                  f"{args.convnext_model_name_or_path}. "
                  "Falling back to default ImageNet normalization stats. "
                  f"Original error: {exc}")

        normalizer = DINOInputNormalizer(
            mean=image_mean,
            std=image_std,
            enabled=not args.disable_encoder_norm,
        )

        mask2former_config = Mask2FormerConfig.from_pretrained(
            args.mask2former_pretrained_model_name_or_path,
            num_labels=args.num_classes,
            id2label=id2label,
            label2id=label2id,
            local_files_only=local_files_only,
        )
        default_feature_strides = list(
            getattr(mask2former_config, "feature_strides", [4, 8, 16, 32]))
        if len(default_feature_strides) < len(hidden_sizes):
            default_feature_strides = [
                2**(index + 2) for index in range(len(hidden_sizes))
            ]
        mask2former_config.feature_strides = [
            default_feature_strides[index] for index in args.feature_indices
        ]
        mask2former_config.use_focal_loss = getattr(args, "use_focal_loss",
                                                    True)
        mask2former_config.focal_alpha = getattr(args, "focal_alpha", 0.25)
        mask2former_config.focal_class_alphas = getattr(
            args, "focal_class_alphas", None)
        mask2former_config.focal_no_object_alpha = getattr(
            args, "focal_no_object_alpha", None)
        mask2former_config.focal_gamma = getattr(args, "focal_gamma", 2.0)
        mask2former_config.focal_normalize_by_num_masks = getattr(
            args, "focal_normalize_by_num_masks", True)
        mask2former_config.classification_loss_type = getattr(
            args, "classification_loss_type", None)
        mask2former_config.seesaw_p = getattr(args, "seesaw_p", 0.8)
        mask2former_config.seesaw_q = getattr(args, "seesaw_q", 2.0)
        mask2former_config.seesaw_eps = getattr(args, "seesaw_eps", 1e-2)

        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            args.mask2former_pretrained_model_name_or_path,
            config=mask2former_config,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )

        pretrained_pixel_decoder_state = (
            self.mask2former.model.pixel_level_module.decoder.state_dict())

        new_pixel_level_module = ConvNeXtPixelLevelModuleBoosted(
            config=self.mask2former.config,
            encoder=encoder,
            normalizer=normalizer,
            feature_indices=args.feature_indices,
            encoder_hidden_sizes=encoder_hidden_sizes,
        )
        load_matching_state_dict(
            new_pixel_level_module.decoder,
            pretrained_pixel_decoder_state,
            module_name="pixel decoder",
        )
        self.mask2former.model.pixel_level_module = new_pixel_level_module

        if args.freeze_mask2former_decoder:
            for name, parameter in self.mask2former.named_parameters():
                if not name.startswith("model.pixel_level_module."):
                    parameter.requires_grad = False

    def forward(self, **kwargs):
        return self.mask2former(**kwargs)

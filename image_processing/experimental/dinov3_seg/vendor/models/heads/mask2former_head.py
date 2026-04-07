# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, Optional, Tuple

from torch import nn
from torch.nn import functional as F

from .pixel_decoder import MSDeformAttnPixelDecoder
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],  # ShapeSpec: [channels, height, width, stride]
        hidden_dim: int = 256,
        num_classes: int = 150,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str = "multi_scale_pixel_decoder",
        mask_dim: Optional[int] = None,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        predictor_nheads: int = 8,
        predictor_dim_feedforward: int = 2048,
        predictor_dec_layers: int = 9,
        predictor_pre_norm: bool = False,
        predictor_enforce_input_project: bool = False,
        common_stride: int = 4,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        orig_input_shape = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]
        if mask_dim is None:
            mask_dim = hidden_dim

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=orig_input_shape,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            conv_dim=hidden_dim,
            mask_dim=mask_dim,
            norm="GN",
            # Match the usual Mask2Former pattern: use the last 3 feature levels
            # for the deformable-attention encoder and keep the highest-resolution
            # feature for the top-down FPN merge.
            transformer_in_features=["2", "3", "4"],
            common_stride=common_stride,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=100,
            nheads=predictor_nheads,
            dim_feedforward=predictor_dim_feedforward,
            dec_layers=predictor_dec_layers,
            pre_norm=predictor_pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=predictor_enforce_input_project,
        )

        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

    def forward_features(self, features, mask=None):
        return self.layers(features, mask)

    def forward(self, features, mask=None):
        output = self.forward_features(features, mask)
        return output

    def predict(self, features, mask=None, rescale_to=(512, 512)):
        output = self.forward_features(features, mask)
        output["pred_masks"] = F.interpolate(
            output["pred_masks"],
            size=rescale_to,
            mode="bilinear",
            align_corners=False,
        )
        return output

    def layers(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions

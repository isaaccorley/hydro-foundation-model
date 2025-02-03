from typing import Sequence, Callable
from functools import partial
import os

import timm
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.model import SegmentationModel
from timm.layers import LayerNorm2d
from einops import rearrange

import torchgeo.models  #swin_v2_b, Swin_V2_B_Weights

from .swin_transformer_v2 import SwinTransformerV2
from .load_pretrained import swin_v2

class SwinBackbone(torch.nn.Module):
    """Mostly from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/backbones.py#L4"""

    def __init__(self, backbone, num_channels: int = 9):
        super(SwinBackbone, self).__init__()
        self.backbone = backbone
        self.out_channels = [
            [4, 128],
            [8, 256],
            [16, 512],
            [32, 1024],
        ]

        if num_channels != 9:
            self.backbone.features[0][0] = torch.nn.Conv2d(num_channels, self.backbone.features[0][0].out_channels, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]



class SwinV2UNet(SegmentationModel):
    def __init__(
        self,
        encoder: str,
        in_channels: int = 3,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        decoder_attention_type: str | None = None,
        classes: int = 1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        norm: Callable = partial(LayerNorm2d, eps=1e-6),
    ):
        super().__init__()
        if encoder == "swinv2-hydro":
            self.encoder, _, _ = swin_v2(encoder_weights)
            self.encoder_channels = self.encoder.dims[:4]
        elif encoder == "swinv2-satlas":
            if in_channels == 3:
                model = torchgeo.models.swin_v2_b(torchgeo.models.Swin_V2_B_Weights.SENTINEL2_SI_RGB_SATLAS)
                self.encoder = SwinBackbone(model, 12)
                self.encoder_channels = [val[1] for val in self.encoder.out_channels]
            else:
                model = torchgeo.models.swin_v2_b(torchgeo.models.Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS)
                self.encoder = SwinBackbone(model, 12)
                self.encoder_channels = [val[1] for val in self.encoder.out_channels]
        else:
            self.encoder = timm.create_model(
                encoder,
                pretrained=True if encoder_weights is not None else False,
                in_chans=in_channels,
                features_only=True,
            )
            self.encoder_channels = self.encoder.feature_info.channels()

        encoder_depth = 4
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(channels, channels, kernel_size=1),
                    norm(channels),
                )
                for channels in self.encoder_channels
            ]
        )
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=[in_channels] + self.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=1,
        )
        self.classification_head = None
        self.name = "u-swinv2-unet"
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, SwinTransformerV2):
            features = self.encoder.get_intermediate_layers(
                x, n=(0, 1, 2, 3), reshape=True
            )
        elif isinstance(self.encoder, SwinBackbone):
            features = self.encoder(x)
        else:
            features = self.encoder(x)
            features = [rearrange(feat, "b h w c -> b c h w") for feat in features]

        features = [up(feat) for feat, up in zip(features, self.upsample)]
        features = [x] + features
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks

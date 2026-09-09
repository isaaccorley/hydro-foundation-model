# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer_v2 import SwinTransformerV2
from .simmim import build_simmim


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    if is_pretrain:
        model = build_simmim(config)
        return model
    elif model_type == "swinv2":
        model = SwinTransformerV2(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWINV2.PATCH_SIZE,
            in_chans=config.MODEL.SWINV2.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWINV2.EMBED_DIM,
            depths=config.MODEL.SWINV2.DEPTHS,
            num_heads=config.MODEL.SWINV2.NUM_HEADS,
            window_size=config.MODEL.SWINV2.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
            qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWINV2.APE,
            patch_norm=config.MODEL.SWINV2.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

import logging
import os
from argparse import Namespace

import torch
import torchvision.transforms as T

from .build import build_model
from ..pretrain.config import get_config
from ..pretrain.utils import load_pretrained, load_checkpoint


def transforms(config):
    mean = torch.tensor(config.DATA.MEAN) / 10000.0
    std = torch.tensor(config.DATA.STD) / 10000.0

    if config.DATA.BANDS == "rgb":
        mean = mean[[3, 2, 1]]
        std = std[[3, 2, 1]]

    return T.Compose(
        [
            T.Lambda(lambda x: x / 10000.0),
            T.Normalize(mean=mean, std=std),
        ]
    )


def swin_v2(config_path):
    logger = logging.getLogger()
    os.environ["LOCAL_RANK"] = "-1"

    args = Namespace(cfg=config_path, opts=None)
    cfg = get_config(args)

    model = build_model(cfg, is_pretrain=False)
    load_pretrained(cfg, model, logger)

    tfrms = transforms(cfg)

    return model, tfrms, cfg


def simmim(config_path):
    logger = logging.getLogger()
    os.environ["LOCAL_RANK"] = "-1"

    args = Namespace(cfg=config_path, opts=None)
    cfg = get_config(args)

    model = build_model(cfg, is_pretrain=True)
    _ = load_checkpoint(config=cfg, model=model, logger=logger)

    tfrms = transforms(cfg)

    return model, tfrms, cfg

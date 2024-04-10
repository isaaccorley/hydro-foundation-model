import logging
import os
import sys
from argparse import Namespace

import torch
import torchvision.transforms as T


MEAN = torch.tensor([340.76769064, 429.9430203, 614.21682446, 590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375, 2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131])
STD = torch.tensor([554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])


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
    sys.path.append("./Swin-Transformer/")
    from config import get_config
    from models import build_model
    from utils_simmim import load_pretrained

    logger = logging.getLogger()
    os.environ["LOCAL_RANK"] = "1"

    args = Namespace(cfg=config_path, opts=None)
    cfg = get_config(args)

    model = build_model(cfg, is_pretrain=False)
    load_pretrained(cfg, model, logger)

    tfrms = transforms(cfg)

    return model, tfrms, cfg

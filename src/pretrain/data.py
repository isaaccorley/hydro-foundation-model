# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import os
import glob

import numpy as np
import torch
import rasterio
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate


class HydroDataset(torch.utils.data.Dataset):
    BAND_SETS = ("rgb", "all")

    def __init__(self, root, bands, transforms=None, ext=".tif"):
        assert bands in self.BAND_SETS
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.images = glob.glob(os.path.join(root, f"*{ext}"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with rasterio.open(path) as f:
            x = f.read().astype(np.float32)
        x = torch.from_numpy(x).to(torch.float)

        if self.bands == "rgb":
            x = x[[3, 2, 1], ...]

        if self.transforms is not None:
            x = self.transforms(x)
        return x, 0


class MaskGenerator:
    def __init__(
        self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self, config):
        mean = torch.tensor(config.DATA.MEAN) / 10000.0
        std = torch.tensor(config.DATA.STD) / 10000.0

        if config.DATA.BANDS == "rgb":
            mean = mean[[3, 2, 1]]
            std = std[[3, 2, 1]]

        self.transform_img = T.Compose(
            [
                T.RandomResizedCrop(
                    config.DATA.IMG_SIZE,
                    scale=(0.67, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.Lambda(lambda x: x / 10000.0),
                T.Normalize(mean=mean, std=std),
            ]
        )

        if config.MODEL.TYPE == "swinv2":
            model_patch_size = config.MODEL.SWINV2.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(
                    default_collate([batch[i][0][item_idx] for i in range(batch_num)])
                )
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config):
    transform = SimMIMTransform(config)
    dataset = HydroDataset(
        root=config.DATA.DATA_PATH, bands=config.DATA.BANDS, transforms=transform
    )
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dataloader

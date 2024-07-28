from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        keys='coco',
        train_all=True,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        # ratio
        if train_all:
            split_0, split_1 = {
                "train": (0.0, 1.0),
                "val": (splits[0], splits[0] + splits[1]),
                "test": (splits[0] + splits[1], 1.0),
            }[split]
        else:
            split_0, split_1 = {
                "train": (0.0, splits[0]),
                "val": (splits[0], splits[0] + splits[1]),
                "test": (splits[0] + splits[1], 1.0),
            }[split]

        self.seeds = []
        for key in keys.split(','):
            anno_path = f'{self.path}/json/seeds_{key}_vanilla.json'
            print(f'Load {anno_path}')
            with open(anno_path, 'r') as f:
                value = json.load(f)
            idx_0 = math.floor(split_0 * len(value))
            idx_1 = math.floor(split_1 * len(value))
            print(f"==> {split} {key}: {idx_1 - idx_0} samples")
            self.seeds += value[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_id, seeds = self.seeds[i]
        seed = seeds[torch.randint(0, len(seeds), ()).item()]

        prompt = seed["edit"]
        image_0_dir = os.path.join(self.path, seed["image_0_dir"]) # removal image
        image_1_dir = os.path.join(self.path, seed["image_1_dir"]) # added image (original image)

        try:
            image_0 = Image.open(image_0_dir).convert("RGB")
            image_1 = Image.open(image_1_dir).convert("RGB")
        except:
            print('cannot load image', image_0_dir, image_1_dir)
            print('resample')
            return self.__getitem__(random.randint(0, len(self) - 1))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        # print(torch.tensor(np.array(image_0)).float().shape, torch.tensor(np.array(image_1)).float().shape, image_0_dir, image_1_dir)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))

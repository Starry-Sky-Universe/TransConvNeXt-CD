"""
TransConvNeXt-CD: Dataset module for LEVIR-CD dataset.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


class LEVIR_Dataset(Dataset):
    """LEVIR-CD change detection dataset."""

    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.t1_paths = sorted(glob.glob(os.path.join(root_dir, mode, "A", "*.png")))
        self.t2_paths = sorted(glob.glob(os.path.join(root_dir, mode, "B", "*.png")))
        self.label_paths = sorted(
            glob.glob(os.path.join(root_dir, mode, "label", "*.png"))
        )

        if mode == "train":
            self.aug = A.Compose(
                [
                    A.RandomCrop(
                        height=CFG.train_img_size, width=CFG.train_img_size, p=1.0
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    A.OneOf(
                        [
                            A.ElasticTransform(
                                alpha=120,
                                sigma=120 * 0.05,
                                alpha_affine=120 * 0.03,
                                p=0.5,
                            ),
                            A.GridDistortion(p=0.5),
                            A.OpticalDistortion(
                                distort_limit=1, shift_limit=0.5, p=0.5
                            ),
                        ],
                        p=0.3,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.HueSaturationValue(p=0.5),
                        ],
                        p=0.3,
                    ),
                    A.Normalize(),
                    ToTensorV2(),
                ],
                additional_targets={"image_0": "image"},
            )
        else:
            self.aug = A.Compose(
                [A.Normalize(), ToTensorV2()],
                additional_targets={"image_0": "image"},
            )

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        t1 = cv2.imread(self.t1_paths[idx])
        t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2RGB)
        t2 = cv2.imread(self.t2_paths[idx])
        t2 = cv2.cvtColor(t2, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = (label > 128).astype(np.float32)

        augmented = self.aug(image=t1, image_0=t2, mask=label)
        return augmented["image"], augmented["image_0"], augmented["mask"].unsqueeze(0)
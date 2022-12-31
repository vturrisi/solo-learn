# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
from PIL import Image
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
)
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from omegaconf import OmegaConf


def test_transforms():
    cfg = OmegaConf.create(
        {
            "crop_size": 224,
            "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
            "color_jitter": {
                "prob": 0.8,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.4,
                "hue": 0.2,
            },
            "grayscale": {"prob": 0.5},
            "gaussian_blur": {"prob": 0.5},
            "solarization": {"prob": 0.2},
            "equalization": {"prob": 0.0},
            "horizontal_flip": {"prob": 0.5},
            "num_crops": 2,
        }
    )

    im = np.random.rand(100, 100, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")

    cfg.crop_size = 32
    T = build_transform_pipeline("cifar10", cfg)
    assert T(im).size(1) == 32

    cfg.crop_size = 96
    T = build_transform_pipeline("stl10", cfg)
    assert T(im).size(1) == 96

    cfg.crop_size = 224
    T = build_transform_pipeline("imagenet100", cfg)
    assert T(im).size(1) == 224

    num_large_crops = 10
    assert (
        len(prepare_n_crop_transform([T], num_crops_per_aug=[num_large_crops])(im))
        == num_large_crops
    )

    cfg = OmegaConf.create(
        {
            "crop_size": 224,
            "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
            "color_jitter": {
                "prob": 0.8,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.4,
                "hue": 0.2,
            },
            "grayscale": {"prob": 0.5},
            "gaussian_blur": {"prob": 0.5},
            "solarization": {"prob": 0.2},
            "equalization": {"prob": 0.0},
            "horizontal_flip": {"prob": 0.5},
            "num_crops": 2,
        }
    )
    cfg_small = OmegaConf.create(
        {
            "crop_size": 96,
            "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
            "color_jitter": {
                "prob": 0.8,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.4,
                "hue": 0.2,
            },
            "grayscale": {"prob": 0.5},
            "gaussian_blur": {"prob": 0.5},
            "solarization": {"prob": 0.2},
            "equalization": {"prob": 0.0},
            "horizontal_flip": {"prob": 0.5},
            "num_crops": 6,
        }
    )

    pipelines = []
    for aug_cfg in [cfg, cfg_small]:
        pipelines.append(
            NCropAugmentation(build_transform_pipeline("imagenet100", aug_cfg), aug_cfg.num_crops)
        )
    transform = FullTransformPipeline(pipelines)
    crops = transform(im)
    sizes = [224] * 2 + [96] * 6
    for crop, size in zip(crops, sizes):
        assert crop.size(1) == size


def test_data():
    kwargs = dict(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.2,
        gaussian_prob=0.5,
        solarization_prob=0.4,
    )

    cfg = OmegaConf.create(
        {
            "crop_size": 32,
            "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
            "color_jitter": {
                "prob": 0.8,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.4,
                "hue": 0.2,
            },
            "grayscale": {"prob": 0.5},
            "gaussian_blur": {"prob": 0.5},
            "solarization": {"prob": 0.2},
            "equalization": {"prob": 0.0},
            "horizontal_flip": {"prob": 0.5},
            "num_crops": 2,
        }
    )
    pipelines = []
    for aug_cfg in [cfg]:
        pipelines.append(
            NCropAugmentation(build_transform_pipeline("imagenet100", aug_cfg), aug_cfg.num_crops)
        )
    transform = FullTransformPipeline(pipelines)

    train_dataset = prepare_datasets("cifar10", transform, train_data_path=None)

    assert isinstance(train_dataset, CIFAR10)
    assert len(train_dataset[0]) == 3

    bs = 64
    num_samples_train = len(train_dataset)
    num_batches_train = num_samples_train // bs

    train_loader = prepare_dataloader(train_dataset, batch_size=bs, num_workers=0)

    assert isinstance(train_loader, DataLoader)
    assert num_batches_train == len(train_loader)

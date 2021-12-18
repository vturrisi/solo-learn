# Copyright 2021 solo-learn development team.

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
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10


def test_transforms():
    kwargs = dict(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.2,
        gaussian_prob=0.5,
        solarization_prob=0.4,
        min_scale=0.08,
        max_scale=1.0,
    )

    im = np.random.rand(100, 100, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")

    T = prepare_transform("cifar10", crop_size=32, **kwargs)
    assert T(im).size(1) == 32

    T = prepare_transform("stl10", crop_size=96, **kwargs)
    assert T(im).size(1) == 96

    T = prepare_transform("imagenet100", **kwargs)
    assert T(im).size(1) == 224

    num_large_crops = 10
    assert (
        len(prepare_n_crop_transform([T], num_crops_per_aug=[num_large_crops])(im))
        == num_large_crops
    )

    kwargs_small = dict(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.2,
        gaussian_prob=0.5,
        solarization_prob=0.4,
        crop_size=96,
        min_scale=0.08,
        max_scale=1.0,
    )

    T = [prepare_transform("imagenet100", **kw) for kw in [kwargs, kwargs_small]]
    T = prepare_n_crop_transform(T, num_crops_per_aug=[2, 6])
    crops = T(im)
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

    T = [prepare_transform("cifar10", **kwargs)]
    T = prepare_n_crop_transform(T, num_crops_per_aug=[2])
    train_dataset = prepare_datasets("cifar10", T, data_dir=None)

    assert isinstance(train_dataset, CIFAR10)
    assert len(train_dataset[0]) == 3

    bs = 64
    num_samples_train = len(train_dataset)
    num_batches_train = num_samples_train // bs

    train_loader = prepare_dataloader(train_dataset, batch_size=bs, num_workers=0)

    assert isinstance(train_loader, DataLoader)
    assert num_batches_train == len(train_loader)

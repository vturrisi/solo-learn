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
import torch
from PIL import Image
from solo.utils.pretrain_dataloader import (
    dataset_with_index,
    prepare_dataloader,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

DATA_KWARGS = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_prob": 0.5,
    "solarization_prob": 0.5,
}


def gen_base_kwargs(
    cifar=False, momentum=False, multicrop=False, num_crops=2, num_small_crops=0, batch_size=32
):
    BASE_KWARGS = {
        "encoder": "resnet18",
        "num_classes": 10 if cifar else 100,
        "no_labels": False,
        "backbone_args": {
            "zero_init_residual": True,
            "cifar": cifar,
        },
        "max_epochs": 2,
        "optimizer": "sgd",
        "lars": True,
        "lr": 0.01,
        "grad_clip_lars": True,
        "weight_decay": 0.00001,
        "classifier_lr": 0.5,
        "exclude_bias_n_norm": True,
        "accumulate_grad_batches": 1,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "warmup_cosine",
        "min_lr": 0.0,
        "warmup_start_lr": 0.0,
        "warmup_epochs": 10,
        "multicrop": multicrop,
        "num_crops": num_crops,
        "num_small_crops": num_small_crops,
        "eta_lars": 0.02,
        "lr_decay_steps": None,
        "dali_device": "gpu",
        "batch_size": batch_size,
        "num_workers": 4,
        "data_dir": "/data/datasets",
        "train_dir": "cifar10/train",
        "val_dir": "cifar10/val",
    }
    if momentum:
        BASE_KWARGS["base_tau_momentum"] = 0.99
        BASE_KWARGS["final_tau_momentum"] = 1.0
    return BASE_KWARGS


def gen_batch(b, num_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")
    T = prepare_transform(dataset, multicrop=False, **DATA_KWARGS)
    T = prepare_n_crop_transform(T, num_crops=2)
    x1, x2 = T(im)
    x1 = x1.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)
    x2 = x2.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)

    idx = torch.arange(b)
    label = torch.randint(low=0, high=num_classes, size=(b,))

    batch, batch_idx = [idx, (x1, x2), label], 1

    return batch, batch_idx


def gen_classification_batch(b, num_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")
    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    x = T(im)
    x = x.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)

    label = torch.randint(low=0, high=num_classes, size=(b,))

    batch, batch_idx = (x, label), 1

    return batch, batch_idx


def prepare_dummy_dataloaders(
    dataset, num_crops, num_classes, multicrop=False, num_small_crops=0, batch_size=2
):
    T = prepare_transform(dataset, multicrop=multicrop, **DATA_KWARGS)
    if multicrop:
        size_crops = [224, 96] if dataset == "imagenet100" else [32, 24]
        T = prepare_multicrop_transform(
            T, size_crops=size_crops, num_crops=[num_crops, num_small_crops]
        )
    else:
        T = prepare_n_crop_transform(T, num_crops)
    dataset = dataset_with_index(FakeData)(
        image_size=(3, 224, 224),
        num_classes=num_classes,
        transform=T,
        size=1024,
    )
    train_dl = prepare_dataloader(dataset, batch_size=batch_size, num_workers=0)

    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=num_classes, transform=T_val)
    val_dl = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    return train_dl, val_dl


def prepare_classification_dummy_dataloaders(dataset, num_classes):
    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=num_classes, transform=T_val)
    train_dl = val_dl = DataLoader(dataset, batch_size=2, num_workers=0, drop_last=False)

    return train_dl, val_dl

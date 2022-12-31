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

import inspect

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import Trainer
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    dataset_with_index,
    prepare_dataloader,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData


def gen_base_cfg(
    method_name, batch_size, num_classes, num_large_crops=2, num_small_crops=0, momentum=False
):
    cfg = {
        "name": "test",
        "method": method_name,
        "backbone": {"name": "resnet18"},
        "data": {
            "dataset": "custom",
            "train_path": ".",
            "val_path": ".",
            "format": "image_folder",
            "num_workers": 4,
            "num_large_crops": num_large_crops,
            "num_small_crops": num_small_crops,
            "num_classes": num_classes,
        },
        "optimizer": {
            "name": "lars",
            "batch_size": batch_size,
            "lr": 0.3,
            "classifier_lr": 0.1,
            "weight_decay": 1e-5,
            "kwargs": {"momentum": 0.9},
        },
        "scheduler": {"name": "warmup_cosine"},
        "checkpoint": {"enabled": False},
        "auto_resume": {"enabled": False},
        "max_epochs": 5,
        "devices": 1,
        "accelerator": "cpu",
        "num_nodes": 1,
    }
    if momentum:
        cfg["momentum"] = {"base_tau": 0.99, "final_tau": 1.0}
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def gen_trainer(cfg, callbacks=None):
    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, (list, tuple)):
        callbacks = [callbacks]

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update({"logger": None, "enable_checkpointing": False, "fast_dev_run": True})
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks)
    return trainer


def gen_batch(b, num_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    x1 = torch.randn(b, 3, size, size, requires_grad=True)
    x2 = torch.randn(b, 3, size, size, requires_grad=True)

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
    dataset, num_large_crops, num_classes, num_small_crops=0, batch_size=2
):
    is_cifar = "cifar" in dataset

    cfg = {
        "augmentations": [
            {
                "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.8,
                    "contrast": 0.8,
                    "saturation": 0.8,
                    "hue": 0.2,
                    "prob": 0.8,
                },
                "grayscale": {
                    "enabled": True,
                    "prob": 0.2,
                },
                "gaussian_blur": {
                    "enabled": True,
                    "prob": 0.2,
                },
                "solarization": {
                    "enabled": True,
                    "prob": 0.2,
                },
                "equalization": {
                    "enabled": False,
                    "prob": 0,
                },
                "horizontal_flip": {
                    "enabled": True,
                    "prob": 0.5,
                },
                "crop_size": 224 if not is_cifar else 32,
                "num_crops": num_large_crops,
            },
        ],
    }

    if num_small_crops > 0:
        small_crop_aug = cfg["augmentations"][0].copy()
        small_crop_aug["crop_size"] = 96 if not is_cifar else 24
        small_crop_aug["num_crops"] = num_small_crops
        cfg["augmentations"].append(small_crop_aug)

    cfg = OmegaConf.create(cfg)

    pipelines = []
    for aug_cfg in cfg.augmentations:
        pipelines.append(
            NCropAugmentation(build_transform_pipeline(dataset, aug_cfg), aug_cfg.num_crops)
        )
    transform = FullTransformPipeline(pipelines)

    dataset = dataset_with_index(FakeData)(
        image_size=(3, 224, 224),
        num_classes=num_classes,
        transform=transform,
        size=1024,
    )
    train_dl = prepare_dataloader(dataset, batch_size=batch_size, num_workers=0)

    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.Resize(224) if not is_cifar else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=num_classes, transform=T_val)
    val_dl = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    return train_dl, val_dl


def prepare_classification_dummy_dataloaders(dataset, num_classes):
    is_cifar = "cifar" in dataset
    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.Resize(224) if not is_cifar else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=num_classes, transform=T_val)
    train_dl = val_dl = DataLoader(dataset, batch_size=2, num_workers=0, drop_last=False)

    return train_dl, val_dl

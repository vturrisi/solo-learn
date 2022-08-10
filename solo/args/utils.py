# Copyright 2022 solo-learn development team.

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

import os
from argparse import Namespace
from contextlib import suppress

N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}


def additional_setup_pretrain(args: Namespace):
    """Provides final setup for pretraining to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, create
    transformations kwargs, correctly parse gpus, identify if a cifar dataset
    is being used and adjust the lr.

    Args:
        args (Namespace): object that needs to contain, at least:
        - dataset: dataset name.
        - brightness, contrast, saturation, hue, min_scale: required augmentations
            settings.
        - dali: flag to use dali.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.

        [optional]
        - gaussian_prob, solarization_prob: optional augmentations settings.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(args.train_data_path) if entry.is_dir]),
        )

    unique_augs = max(
        len(p)
        for p in [
            args.brightness,
            args.contrast,
            args.saturation,
            args.hue,
            args.color_jitter_prob,
            args.gray_scale_prob,
            args.horizontal_flip_prob,
            args.gaussian_prob,
            args.solarization_prob,
            args.equalization_prob,
            args.crop_size,
            args.min_scale,
            args.max_scale,
        ]
    )
    assert len(args.num_crops_per_aug) == unique_augs

    # assert that either all unique augmentation pipelines have a unique
    # parameter or that a single parameter is replicated to all pipelines
    for p in [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "color_jitter_prob",
        "gray_scale_prob",
        "horizontal_flip_prob",
        "gaussian_prob",
        "solarization_prob",
        "equalization_prob",
        "crop_size",
        "min_scale",
        "max_scale",
    ]:
        values = getattr(args, p)
        n = len(values)
        assert n == unique_augs or n == 1

        if n == 1:
            setattr(args, p, getattr(args, p) * unique_augs)

    args.unique_augs = unique_augs

    if unique_augs > 1:
        args.transform_kwargs = [
            dict(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                color_jitter_prob=color_jitter_prob,
                gray_scale_prob=gray_scale_prob,
                horizontal_flip_prob=horizontal_flip_prob,
                gaussian_prob=gaussian_prob,
                solarization_prob=solarization_prob,
                equalization_prob=equalization_prob,
                crop_size=crop_size,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            for (
                brightness,
                contrast,
                saturation,
                hue,
                color_jitter_prob,
                gray_scale_prob,
                horizontal_flip_prob,
                gaussian_prob,
                solarization_prob,
                equalization_prob,
                crop_size,
                min_scale,
                max_scale,
            ) in zip(
                args.brightness,
                args.contrast,
                args.saturation,
                args.hue,
                args.color_jitter_prob,
                args.gray_scale_prob,
                args.horizontal_flip_prob,
                args.gaussian_prob,
                args.solarization_prob,
                args.equalization_prob,
                args.crop_size,
                args.min_scale,
                args.max_scale,
            )
        ]

        # find number of big/small crops
        big_size = args.crop_size[0]
        num_large_crops = num_small_crops = 0
        for size, n_crops in zip(args.crop_size, args.num_crops_per_aug):
            if big_size == size:
                num_large_crops += n_crops
            else:
                num_small_crops += n_crops
        args.num_large_crops = num_large_crops
        args.num_small_crops = num_small_crops
    else:
        args.transform_kwargs = dict(
            brightness=args.brightness[0],
            contrast=args.contrast[0],
            saturation=args.saturation[0],
            hue=args.hue[0],
            color_jitter_prob=args.color_jitter_prob[0],
            gray_scale_prob=args.gray_scale_prob[0],
            horizontal_flip_prob=args.horizontal_flip_prob[0],
            gaussian_prob=args.gaussian_prob[0],
            solarization_prob=args.solarization_prob[0],
            equalization_prob=args.equalization_prob[0],
            crop_size=args.crop_size[0],
            min_scale=args.min_scale[0],
            max_scale=args.max_scale[0],
        )

        # find number of big/small crops
        args.num_large_crops = args.num_crops_per_aug[0]
        args.num_small_crops = 0

    # add support for custom mean and std
    if args.dataset == "custom":
        if isinstance(args.transform_kwargs, dict):
            args.transform_kwargs["mean"] = args.mean
            args.transform_kwargs["std"] = args.std
        else:
            for kwargs in args.transform_kwargs:
                kwargs["mean"] = args.mean
                kwargs["std"] = args.std

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}
    if "resnet" in args.backbone and "wide" not in args.backbone:
        args.backbone_args["zero_init_residual"] = args.zero_init_residual
    elif "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.backbone:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.zero_init_residual
    with suppress(AttributeError):
        del args.patch_size

    if args.data_format == "dali":
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
    elif args.optimizer == "lars":
        args.extra_optimizer_args["momentum"] = 0.9
        args.extra_optimizer_args["eta"] = args.eta_lars
        args.extra_optimizer_args["clip_lars_lr"] = args.grad_clip_lars
        args.extra_optimizer_args["exclude_bias_n_norm"] = args.exclude_bias_n_norm_lars
    elif args.optimizer == "adamw":
        args.extra_optimizer_args["betas"] = [args.adamw_beta1, args.adamw_beta2]

    with suppress(AttributeError):
        del args.eta_lars
    with suppress(AttributeError):
        del args.grad_clip_lars
    with suppress(AttributeError):
        del args.exclude_bias_n_norm_lars
    with suppress(AttributeError):
        del args.adamw_beta1
    with suppress(AttributeError):
        del args.adamw_beta2

    if isinstance(args.devices, int):
        args.devices = [args.devices]
    elif isinstance(args.devices, str):
        args.devices = [int(device) for device in args.devices.split(",") if device]

    # adjust lr according to batch size
    try:
        num_nodes = args.num_nodes or 1
    except AttributeError:
        num_nodes = 1

    scale_factor = args.batch_size * len(args.devices) * num_nodes / 256
    args.lr = args.lr * scale_factor
    args.classifier_lr = args.classifier_lr * scale_factor


def additional_setup_linear(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(args.train_data_path) if entry.is_dir]),
        )

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}
    if "resnet" not in args.backbone and "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.backbone:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.patch_size

    if args.data_format == "dali":
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
    elif args.optimizer == "lars":
        args.extra_optimizer_args["momentum"] = 0.9
        args.extra_optimizer_args["exclude_bias_n_norm"] = args.exclude_bias_n_norm_lars
    elif args.optimizer == "adamw":
        args.extra_optimizer_args["betas"] = [args.adamw_beta1, args.adamw_beta2]

    with suppress(AttributeError):
        del args.exclude_bias_n_norm_lars
    with suppress(AttributeError):
        del args.adamw_beta1
    with suppress(AttributeError):
        del args.adamw_beta2

    if isinstance(args.devices, int):
        args.devices = [args.devices]
    elif isinstance(args.devices, str):
        args.devices = [int(device) for device in args.devices.split(",") if device]

    # adjust lr according to batch size
    try:
        num_nodes = args.num_nodes or 1
    except AttributeError:
        num_nodes = 1

    scale_factor = args.batch_size * len(args.devices) * num_nodes / 256
    args.lr = args.lr * scale_factor

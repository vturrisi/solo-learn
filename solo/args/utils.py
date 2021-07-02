import argparse

N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}


def additional_setup_contrastive(args: argparse.Namespace):
    """
    Provides final setup for contrastive pretrain to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, create
    transformations kwargs, correctly parse gpus, identify if a cifar dataset
    is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
            - dataset: dataset name
            - brightness, contrast, saturation, hue, min_scale_crop: required augmentations settings
            - asymmetric_augmentations: flag to apply asymmetric augmentations
            - multicrop: flag to use multicrop
            - dali: flag to use dali
            - optimizer: optimizer name being used
            - gpus: list of gpus to use
            - lr: learning rate

            [optional]
            - gaussian_prob solarization_prob: optinal augmentations settings

    """

    args.transform_kwargs = {}

    assert args.dataset in N_CLASSES_PER_DATASET
    args.n_classes = N_CLASSES_PER_DATASET[args.dataset]

    if args.asymmetric_augmentations:
        if args.dataset in ["cifar10", "cifar100"]:
            gaussian_probs = [0.0, 0.0]
        else:
            gaussian_probs = [1.0, 0.1]
        solarization_probs = [0.0, 0.2]

        args.transform_kwargs = [
            dict(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
                gaussian_prob=gaussian_probs[0],
                solarization_prob=solarization_probs[0],
                min_scale_crop=args.min_scale_crop,
            ),
            dict(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
                gaussian_prob=gaussian_probs[1],
                solarization_prob=solarization_probs[1],
                min_scale_crop=args.min_scale_crop,
            ),
        ]
    elif not args.multicrop:
        args.transform_kwargs = dict(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_prob,
            solarization_prob=args.solarization_prob,
            min_scale_crop=args.min_scale_crop,
        )
    else:
        args.transform_kwargs = dict(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_prob,
            solarization_prob=args.solarization_prob,
        )

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(s) for s in args.gpus.split(",")]
    # adjust lr according to batch size
    args.lr = args.lr * args.batch_size * len(args.gpus) / 256


def additional_setup_linear(args: argparse.Namespace):
    """
    Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset,
    correctly parse gpus, identify if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
            - dataset: dataset name
            - optimizer: optimizer name being used
            - gpus: list of gpus to use
            - lr: learning rate

    """
    assert args.dataset in N_CLASSES_PER_DATASET
    args.n_classes = N_CLASSES_PER_DATASET[args.dataset]

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(s) for s in args.gpus.split(",")]

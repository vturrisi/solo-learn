import argparse
from solo.args.utils import additional_setup_contrastive, additional_setup_linear


def test_additional_setup_contrastive():
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": True,
        "multicrop": False,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_prob": 0.5,
        "solarization_prob": 0.5,
        "min_scale_crop": 0.08,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_contrastive(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # - asymmetric - multicrop
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": False,
        "multicrop": False,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_prob": 0.5,
        "solarization_prob": 0.5,
        "min_scale_crop": 0.08,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_contrastive(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # + multicrop
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": False,
        "multicrop": True,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_prob": 0.5,
        "solarization_prob": 0.5,
        "min_scale_crop": 0.08,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_contrastive(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # check for different gpu syntax
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": True,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_prob": 0.5,
        "solarization_prob": 0.5,
        "min_scale_crop": 0.08,
        "dali": True,
        "optimizer": "sgd",
        "gpus": 0,
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_contrastive(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args


def test_additional_setup_linear():
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": True,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_linear(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)

    # check for different gpu syntax
    args = {
        "dataset": "imagenet100",
        "asymmetric_augmentations": True,
        "dali": True,
        "optimizer": "sgd",
        "gpus": 0,
        "lr": 0.1,
        "batch_size": 128,
    }
    args = argparse.Namespace(**args)

    additional_setup_linear(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)

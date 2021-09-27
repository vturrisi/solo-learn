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

import argparse
import os
import subprocess
import textwrap
from pathlib import Path

from solo.args.utils import additional_setup_linear, additional_setup_pretrain
from tests.dali.utils import DummyDataset


def test_setup_pretrain():

    dummy_script = """
        from solo.args.setup import parse_args_pretrain

        args = parse_args_pretrain()
    """
    dummy_script = textwrap.dedent(dummy_script)

    dummy_args = [
        "--dataset",
        "cifar10",
        "--encoder",
        "resnet18",
        "--data_dir",
        "./datasets",
        "--max_epochs",
        "1000",
        "--gpus",
        "0",
        "--precision",
        "16",
        "--optimizer",
        "sgd",
        "--lars",
        "--exclude_bias_n_norm",
        "--scheduler",
        "warmup_cosine",
        "--lr",
        "1.0",
        "--classifier_lr",
        "0.1",
        "--weight_decay",
        "1e-5",
        "--batch_size",
        "256",
        "--num_workers",
        "5",
        "--brightness",
        "0.4",
        "--contrast",
        "0.4",
        "--saturation",
        "0.2",
        "--hue",
        "0.1",
        "--gaussian_prob",
        "0.0",
        "--solarization_prob",
        "0.2",
        "--name",
        "test",
        "--project",
        "solo-learn",
        "--entity",
        "unitn-mhug",
        "--wandb",
        "--save_checkpoint",
        "--method",
        "byol",
        "--proj_output_dim",
        "256",
        "--proj_hidden_dim",
        "4096",
        "--pred_hidden_dim",
        "4096",
        "--base_tau_momentum",
        "0.99",
        "--final_tau_momentum",
        "1.0",
        "--momentum_classifier",
        "--wandb",
        "--save_checkpoint",
        "--auto_umap",
    ]
    # Write string to a file
    with open("dummy_script.py", "w") as f:
        f.write(dummy_script)

    # Run the python file as a separate process
    try:
        script = ["python3", "dummy_script.py"] + dummy_args
        subprocess.check_output(script)
        worked = True
    except subprocess.CalledProcessError as e:
        print("error code", e.returncode, e.output)
        worked = False
    assert worked

    try:
        os.remove("dummy_script.py")
    except:
        pass


def test_setup_linear():

    dummy_script = """
        from solo.args.setup import parse_args_linear

        args = parse_args_linear()
    """
    dummy_script = textwrap.dedent(dummy_script)

    dummy_args = [
        "--dataset",
        "imagenet100",
        "--encoder",
        "resnet18",
        "--data_dir",
        "/datasets",
        "--train_dir",
        "imagenet-100/train",
        "--val_dir",
        "imagenet-100/val",
        "--max_epochs",
        "100",
        "--gpus",
        "0",
        "--distributed_backend",
        "ddp",
        "--sync_batchnorm",
        "--precision",
        "16",
        "--optimizer",
        "sgd",
        "--scheduler",
        "step",
        "--lr",
        "3.0",
        "--lr_decay_steps",
        "60",
        "--weight_decay",
        "0",
        "--batch_size",
        "128",
        "--num_workers",
        "10",
        "--dali",
        "--name",
        "test",
        "--pretrained_feature_extractor",
        "PATH",
        "--project",
        "solo-learn",
        "--wandb",
        "--save_checkpoint",
    ]
    # Write string to a file
    with open("dummy_script.py", "w") as f:
        f.write(dummy_script)

    # Run the python file as a separate process
    try:
        script = ["python3", "dummy_script.py"] + dummy_args
        subprocess.check_output(script)
        worked = True
    except subprocess.CalledProcessError as e:
        print("error code", e.returncode, e.output)
        worked = False
    assert worked

    try:
        os.remove("dummy_script.py")
    except:
        pass


def test_additional_setup_pretrain():
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "multicrop": False,
        "brightness": [0.4],
        "contrast": [0.4],
        "saturation": [0.2],
        "hue": [0.1],
        "gaussian_prob": [1.0, 0.1],
        "solarization_prob": [0.2, 0.1],
        "min_scale": [0.08],
        "size": [224],
        "num_crops": 2,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # - asymmetric - multicrop
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "multicrop": False,
        "brightness": [0.4],
        "contrast": [0.4],
        "saturation": [0.2],
        "hue": [0.1],
        "gaussian_prob": [0.5],
        "solarization_prob": [0.5],
        "min_scale": [0.08],
        "size": [224],
        "num_crops": 2,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # + multicrop
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "multicrop": True,
        "brightness": [0.4],
        "contrast": [0.4],
        "saturation": [0.2],
        "hue": [0.1],
        "gaussian_prob": [0.5],
        "solarization_prob": [0.5],
        "min_scale": [0.08],
        "size": [224],
        "num_crops": 2,
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # check for different gpu syntax
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "multicrop": False,
        "brightness": [0.4],
        "contrast": [0.4],
        "saturation": [0.2],
        "hue": [0.1],
        "gaussian_prob": [0.5, 0.2],
        "solarization_prob": [0.5, 0.3],
        "min_scale": [0.08],
        "size": [224],
        "num_crops": 2,
        "dali": True,
        "optimizer": "sgd",
        "gpus": 0,
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # check for different encoder / custom dataset
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        args = {
            "encoder": "vit_small",
            "dataset": "custom",
            "data_dir": Path("."),
            "train_dir": "dummy_train",
            "val_dir": "dummy_val",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.228, 0.224, 0.225],
            "multicrop": False,
            "brightness": [0.4],
            "contrast": [0.4],
            "saturation": [0.2],
            "hue": [0.1],
            "gaussian_prob": [0.5, 0.2],
            "solarization_prob": [0.5, 0.3],
            "min_scale": [0.08],
            "size": [224],
            "num_crops": 2,
            "dali": True,
            "optimizer": "sgd",
            "gpus": 0,
            "lr": 0.1,
            "batch_size": 128,
            "patch_size": 16,
        }
        args = argparse.Namespace(**args)

        additional_setup_pretrain(args)

        assert args.backbone_args["cifar"] is False
        assert "momentum" in args.extra_optimizer_args
        assert isinstance(args.gpus, list)
        assert "transform_kwargs" in args


def test_additional_setup_linear():
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "dali": True,
        "optimizer": "sgd",
        "gpus": "0,1",
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_linear(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)

    # check for different gpu syntax
    args = {
        "encoder": "resnet18",
        "dataset": "imagenet100",
        "dali": True,
        "optimizer": "sgd",
        "gpus": 0,
        "lr": 0.1,
        "batch_size": 128,
        "zero_init_residual": False,
    }
    args = argparse.Namespace(**args)

    additional_setup_linear(args)

    assert args.backbone_args["cifar"] is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)

    # check for different encoder / custom dataset
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        args = {
            "encoder": "vit_small",
            "dataset": "custom",
            "data_dir": Path("."),
            "train_dir": "dummy_train",
            "val_dir": "dummy_val",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.228, 0.224, 0.225],
            "size": [224],
            "dali": True,
            "encoder": "vit_small",
            "optimizer": "sgd",
            "gpus": 0,
            "lr": 0.1,
            "batch_size": 128,
            "zero_init_residual": False,
            "patch_size": 16,
        }
        args = argparse.Namespace(**args)

        additional_setup_linear(args)

        assert args.backbone_args["cifar"] is False
        assert "momentum" in args.extra_optimizer_args
        assert isinstance(args.gpus, list)

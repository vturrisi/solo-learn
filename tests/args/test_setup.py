import argparse
import os
import subprocess
import textwrap
from solo.args.utils import additional_setup_pretrain, additional_setup_linear


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
        "0.0 0.0",
        "--solarization_prob",
        "0.0 0.2",
        "--name",
        "test",
        "--project",
        "solo-learn",
        "--entity",
        "unitn-mhug",
        "--wandb",
        "--method",
        "byol",
        "--output_dim",
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
    ]
    # Write string to a file
    with open("dummy_script.py", "w") as f:
        f.write(dummy_script)

    # Run the python file as a separate process
    try:
        script = ["python3", "dummy_script.py"] + dummy_args
        subprocess.check_output(script)
    except subprocess.CalledProcessError as e:
        print("error code", e.returncode, e.output)

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
        "60 80",
        "--weight_decay",
        "0",
        "--batch_size",
        "128",
        "--num_workers",
        "10",
        "--dali",
        "--name",
        "test",
        "--pretrained_feature_extractor" "PATH",
        "--project",
        "solo-learn",
        "--wandb",
    ]
    # Write string to a file
    with open("dummy_script.py", "w") as f:
        f.write(dummy_script)

    # Run the python file as a separate process
    try:
        script = ["python3", "dummy_script.py"] + dummy_args
        subprocess.check_output(script)
    except subprocess.CalledProcessError as e:
        print("error code", e.returncode, e.output)

    try:
        os.remove("dummy_script.py")
    except:
        pass


def test_additional_setup_pretrain():
    args = {
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
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # - asymmetric - multicrop
    args = {
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
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # + multicrop
    args = {
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
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args

    # check for different gpu syntax
    args = {
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
    }
    args = argparse.Namespace(**args)

    additional_setup_pretrain(args)

    assert args.cifar is False
    assert "momentum" in args.extra_optimizer_args
    assert isinstance(args.gpus, list)
    assert "transform_kwargs" in args


def test_additional_setup_linear():
    args = {
        "dataset": "imagenet100",
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

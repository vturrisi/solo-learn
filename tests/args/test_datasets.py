import argparse
from solo.args.dataset import dataset_args, augmentations_args


def test_argparse_dataset():
    parser = argparse.ArgumentParser()
    dataset_args(parser)
    actions = [vars(action)["dest"] for action in vars(parser)["_actions"]]

    assert "dataset" in actions
    assert "data_dir" in actions
    assert "train_dir" in actions
    assert "val_dir" in actions
    assert "dali" in actions
    assert "dali_device" in actions


def test_argparse_augmentations():
    parser = argparse.ArgumentParser()
    augmentations_args(parser)
    actions = [vars(action)["dest"] for action in vars(parser)["_actions"]]

    assert "multicrop" in actions
    assert "num_crops" in actions
    assert "num_small_crops" in actions

    assert "brightness" in actions
    assert "contrast" in actions
    assert "saturation" in actions
    assert "hue" in actions
    assert "gaussian_prob" in actions
    assert "solarization_prob" in actions
    assert "min_scale" in actions

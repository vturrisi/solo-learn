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

    assert "num_crops_per_aug" in actions

    assert "brightness" in actions
    assert "contrast" in actions
    assert "saturation" in actions
    assert "hue" in actions
    assert "color_jitter_prob" in actions
    assert "gray_scale_prob" in actions
    assert "horizontal_flip_prob" in actions
    assert "gaussian_prob" in actions
    assert "solarization_prob" in actions
    assert "min_scale" in actions

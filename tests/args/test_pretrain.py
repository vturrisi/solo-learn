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

from omegaconf import OmegaConf
from solo.args.pretrain import parse_cfg


def test_pretrain_parse_cfg():
    cfg = {
        "name": "test",
        "method": "barlow_twins",
        "backbone": {"name": "resnet18"},
        "data": {
            "dataset": "imagenet100",
            "train_path": ".",
            "val_path": ".",
            "format": "image_folder",
            "num_workers": 4,
        },
        "augmentations": [
            {
                "crop_size": 224,
                "rrc": {"enabled": True, "crop_min_scale": 0.08, "crop_max_scale": 1.0},
                "color_jitter": {
                    "prob": 0.8,
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "hue": 0.2,
                },
                "grayscale": {"prob": 0.5},
                "gaussian_blur": {"prob": 0.5},
                "solarization": {"prob": 0.1},
                "equalization": {"prob": 0.0},
                "horizontal_flip": {"prob": 0.5},
                "num_crops": 2,
            }
        ],
        "optimizer": {
            "name": "lars",
            "batch_size": 64,
            "lr": 0.3,
            "classifier_lr": 0.1,
            "weight_decay": 1e-5,
        },
        "scheduler": {"name": "warmup_cosine"},
        "checkpoint": {"enabled": False},
        "auto_resume": {"enabled": False},
        "max_epochs": 5,
        "devices": [0],
        "accelerator": "gpu",
        "num_nodes": 1,
    }

    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg = parse_cfg(cfg)

    # assert data config is there
    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")
    assert not OmegaConf.is_missing(cfg, "data.val_path")
    assert not OmegaConf.is_missing(cfg, "data.format")
    assert not OmegaConf.is_missing(cfg, "data.no_labels")
    assert not OmegaConf.is_missing(cfg, "data.fraction")
    assert not OmegaConf.is_missing(cfg, "debug_augmentations")

    # assert lightning config is there
    assert not OmegaConf.is_missing(cfg, "seed")
    assert not OmegaConf.is_missing(cfg, "resume_from_checkpoint")
    assert not OmegaConf.is_missing(cfg, "strategy")

    # assert wandb config is there
    assert not OmegaConf.is_missing(cfg, "wandb")
    assert not OmegaConf.is_missing(cfg, "wandb.enabled")
    assert not OmegaConf.is_missing(cfg, "wandb.entity")
    assert not OmegaConf.is_missing(cfg, "wandb.project")
    assert not OmegaConf.is_missing(cfg, "wandb.offline")

    # extra config done by the parse function
    assert not OmegaConf.is_missing(cfg, "data.num_classes")
    assert not OmegaConf.is_missing(cfg, "data.num_large_crops")
    assert not OmegaConf.is_missing(cfg, "data.num_small_crops")
    assert not OmegaConf.is_missing(cfg, "num_nodes")
    assert not OmegaConf.is_missing(cfg, "optimizer.kwargs")

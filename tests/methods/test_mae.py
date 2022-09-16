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

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from solo.methods import MoCoV3
from solo.methods.mae import MAE

from .utils import DATA_KWARGS, gen_base_kwargs, gen_batch, prepare_dummy_dataloaders


def test_mae():
    method_kwargs = {
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mask_ratio": 0.75,
        "norm_pix_loss": True,
    }

    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, batch_size=2)
    BASE_KWARGS["method"] = "mae"
    BASE_KWARGS["backbone"] = "vit_small"
    BASE_KWARGS["backbone_args"] = {"img_size": 224, "patch_size": 16}

    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = MAE(**kwargs)

    # test arguments
    model.add_and_assert_specific_cfg(cfg)

    # test parameters
    assert model.learnable_params is not None

    # test forward
    batch, _ = gen_batch(cfg.optimizer.batch_size, cfg.data.num_classes, "imagenet100")
    out = model(batch[1][0])
    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (cfg.optimizer.batch_size, cfg.data.num_classes)
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (cfg.optimizer.batch_size, model.features_dim)
    )
    assert (
        "pred" in out
        and isinstance(out["pred"], torch.Tensor)
        and out["pred"].size() == (cfg.optimizer.batch_size, 14 * 14, 16 * 16 * 3)
    )
    assert (
        "mask" in out
        and isinstance(out["mask"], torch.Tensor)
        and out["mask"].size() == (cfg.optimizer.batch_size, 14 * 14)
    )

    # imagenet
    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, batch_size=2)
    BASE_KWARGS["method"] = "mae"
    BASE_KWARGS["backbone"] = "vit_small"
    BASE_KWARGS["backbone_args"] = {"img_size": 224, "patch_size": 16}

    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = MAE(**kwargs)

    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=0,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

    # cifar
    BASE_KWARGS = gen_base_kwargs(cifar=True, momentum=True, batch_size=2)
    BASE_KWARGS["method"] = "mae"
    BASE_KWARGS["backbone"] = "vit_small"
    BASE_KWARGS["backbone_args"] = {"img_size": 32, "patch_size": 16}

    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = MAE(**kwargs)

    trainer = gen_trainer(cfg)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=0,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

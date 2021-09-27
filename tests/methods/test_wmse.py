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

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from solo.methods import WMSE

from .utils import DATA_KWARGS, gen_base_kwargs, gen_batch, prepare_dummy_dataloaders


def test_wmse():
    BASE_KWARGS = gen_base_kwargs(cifar=False, batch_size=8)

    method_kwargs = {
        "proj_hidden_dim": 1024,
        "proj_output_dim": BASE_KWARGS["batch_size"] // 4,
        "whitening_size": BASE_KWARGS["batch_size"] // 2,
        "whitening_iters": 1,
        "whitening_eps": 1e-2,
    }

    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = WMSE(**kwargs, disable_knn_eval=True)

    # test arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    assert model.add_model_specific_args(parser) is not None

    # test parameters
    assert model.learnable_params is not None

    # test forward
    batch, _ = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"], "imagenet100")
    out = model(batch[1][0])
    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"])
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (BASE_KWARGS["batch_size"], model.features_dim)
    )
    assert (
        "v" in out
        and isinstance(out["v"], torch.Tensor)
        and out["v"].size() == (BASE_KWARGS["batch_size"], method_kwargs["proj_output_dim"])
    )

    for num_crops in [2, 4]:
        # imagenet
        BASE_KWARGS = gen_base_kwargs(
            cifar=False, multicrop=False, num_crops=num_crops, batch_size=8
        )
        method_kwargs["output_dim"] = BASE_KWARGS["batch_size"] // 4
        method_kwargs["whitening_size"] = BASE_KWARGS["batch_size"] // 2
        kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
        model = WMSE(**kwargs, disable_knn_eval=True)

        args = argparse.Namespace(**kwargs)
        trainer = Trainer.from_argparse_args(args, fast_dev_run=True)
        train_dl, val_dl = prepare_dummy_dataloaders(
            "imagenet100",
            num_crops=BASE_KWARGS["num_crops"],
            num_small_crops=0,
            num_classes=BASE_KWARGS["num_classes"],
            multicrop=False,
            batch_size=BASE_KWARGS["batch_size"],
        )
        trainer.fit(model, train_dl, val_dl)

        # cifar
        BASE_KWARGS = gen_base_kwargs(
            cifar=False, multicrop=False, num_crops=num_crops, batch_size=8
        )
        method_kwargs["output_dim"] = BASE_KWARGS["batch_size"] // 4
        method_kwargs["whitening_size"] = BASE_KWARGS["batch_size"] // 2
        kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
        model = WMSE(**kwargs, disable_knn_eval=True)

        args = argparse.Namespace(**kwargs)
        trainer = Trainer.from_argparse_args(args, fast_dev_run=True)
        train_dl, val_dl = prepare_dummy_dataloaders(
            "cifar10",
            num_crops=BASE_KWARGS["num_crops"],
            num_small_crops=0,
            num_classes=BASE_KWARGS["num_classes"],
            multicrop=False,
            batch_size=BASE_KWARGS["batch_size"],
        )
        trainer.fit(model, train_dl, val_dl)

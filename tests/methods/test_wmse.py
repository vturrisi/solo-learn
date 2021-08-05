import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from solo.methods import WMSE

from .utils import DATA_KWARGS, gen_base_kwargs, gen_batch, prepare_dummy_dataloaders


def test_wmse():
    BASE_KWARGS = gen_base_kwargs(cifar=False)

    method_kwargs = {
        "proj_hidden_dim": 1024,
        "output_dim": BASE_KWARGS["batch_size"] // 4,
        "whitening_size": BASE_KWARGS["batch_size"] // 2,
        "whitening_iters": 1,
        "whitening_eps": 1e-2,
    }

    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = WMSE(**kwargs)

    batch, batch_idx = gen_batch(
        BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"], "imagenet100"
    )
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    BASE_KWARGS = gen_base_kwargs(cifar=True)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = WMSE(**kwargs)

    batch, batch_idx = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"], "cifar10")
    loss = model.training_step(batch, batch_idx)

    assert loss != 0

    # test arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    assert model.add_model_specific_args(parser) is not None

    # test parameters
    assert model.learnable_params is not None

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
        and out["v"].size() == (BASE_KWARGS["batch_size"], method_kwargs["output_dim"])
    )

    # normal training
    for num_crops in [2, 4]:
        BASE_KWARGS = gen_base_kwargs(cifar=False, multicrop=False, num_crops=num_crops)
        BASE_KWARGS["batch_size"] = 8
        method_kwargs["output_dim"] = BASE_KWARGS["batch_size"] // 4
        method_kwargs["whitening_size"] = BASE_KWARGS["batch_size"] // 2
        kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
        model = WMSE(**kwargs)

        args = argparse.Namespace(**kwargs)
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=False,
            limit_train_batches=2,
            limit_val_batches=2,
        )
        train_dl, val_dl = prepare_dummy_dataloaders(
            "imagenet100",
            num_crops=BASE_KWARGS["num_crops"],
            num_small_crops=0,
            num_classes=BASE_KWARGS["num_classes"],
            multicrop=False,
            batch_size=BASE_KWARGS["batch_size"],
        )
        trainer.fit(model, train_dl, val_dl)

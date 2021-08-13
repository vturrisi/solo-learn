import argparse

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from solo.methods.linear import LinearModel
from torchvision.models import resnet18

from .utils import (
    DATA_KWARGS,
    gen_base_kwargs,
    gen_classification_batch,
    prepare_classification_dummy_dataloaders,
)


def test_linear():
    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, multicrop=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS}
    backbone = resnet18()
    backbone.fc = nn.Identity()
    model = LinearModel(backbone, **kwargs)

    # test arguments
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    assert model.add_model_specific_args(parser) is not None

    batch, batch_idx = gen_classification_batch(
        BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"], "imagenet100"
    )
    out = model(batch[0])

    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"])
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (BASE_KWARGS["batch_size"], model.backbone.inplanes)
    )

    args = argparse.Namespace(**kwargs)
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    train_dl, val_dl = prepare_classification_dummy_dataloaders(
        "imagenet100",
        num_classes=BASE_KWARGS["num_classes"],
    )
    trainer.fit(model, train_dl, val_dl)

    # test optimizers/scheduler
    model.optimizer = "random"
    model.scheduler = "none"
    with pytest.raises(ValueError):
        model.configure_optimizers()

    model.optimizer = "sgd"
    model.scheduler = "none"
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    model.scheduler = "cosine"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    model.scheduler = "reduce"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    model.scheduler = "step"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    model.scheduler = "exponential"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    model.scheduler = "random"
    with pytest.raises(ValueError):
        model.configure_optimizers()

    model.optimizer = "adam"
    model.scheduler = "none"
    model.extra_optimizer_args = {}
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

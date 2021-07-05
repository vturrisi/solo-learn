import argparse

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from solo.methods.linear import LinearModel
from torchvision.models import resnet18

from .utils import (
    DATA_KWARGS,
    gen_base_kwargs,
    prepare_classification_dummy_dataloaders,
    gen_classification_batch,
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
        BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"], "imagenet100"
    )
    out = model(batch[0])

    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (BASE_KWARGS["batch_size"], BASE_KWARGS["n_classes"])
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
        n_classes=BASE_KWARGS["n_classes"],
    )
    trainer.fit(model, train_dl, val_dl)

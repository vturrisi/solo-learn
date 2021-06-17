import argparse

import torch.nn as nn
from pytorch_lightning import Trainer
from solo.methods.linear import LinearModel
from torchvision.models import resnet18

from .utils import DATA_KWARGS, gen_base_kwargs, prepare_classification_dummy_dataloaders


def test_linear():
    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, multicrop=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS}
    backbone = resnet18()
    backbone.fc = nn.Identity()
    model = LinearModel(backbone, **kwargs)
    args = argparse.Namespace(**kwargs)
    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=False, limit_train_batches=2, limit_val_batches=2,
    )
    train_dl, val_dl = prepare_classification_dummy_dataloaders(
        "imagenet100", n_classes=BASE_KWARGS["n_classes"],
    )
    trainer.fit(model, train_dl, val_dl)

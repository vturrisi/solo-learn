from abc import abstractmethod
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def static_lr(get_lr, param_group_indexes, lrs_to_replace):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        zero_init_residual,
        n_classes,
        cifar,
        lr,
        weight_decay,
        max_epochs,
        classifier_lr,
        optimizer,
        lars,
        exclude_bias_n_norm,
        extra_optimizer_args,
        scheduler,
        lr_decay_steps,
        **kwargs
    ):
        super().__init__()

        self.cifar = cifar
        self.zero_init_residual = zero_init_residual
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.classifier_lr = classifier_lr
        self.optimizer = optimizer
        self.lars = lars
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps

        assert encoder in ["resnet18", "resnet50"]
        from torchvision.models import resnet18, resnet50

        self.base_model = {"resnet18": resnet18, "resnet50": resnet50}[encoder]

        # initialize encoder
        self.encoder = self.base_model(zero_init_residual=zero_init_residual)
        self.features_size = self.encoder.inplanes
        # remove fc layer
        self.encoder.fc = nn.Identity()
        if cifar:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.encoder.maxpool = nn.Identity()

        self.classifier = nn.Linear(self.features_size, n_classes)

    @property
    def base_learnable_params(self):
        return [
            {"params": self.encoder.parameters()},
            {
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    @property
    @abstractmethod
    def extra_learnable_params(self):
        pass

    def configure_optimizers(self):

        # collect learnable parameters
        base_learnable_params = list(self.base_learnable_params)
        extra_learnable_params = list(self.extra_learnable_params)
        learnable_params = base_learnable_params + extra_learnable_params
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)

        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=0.003
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer], [scheduler]

    def forward(self, X):
        feat = self.encoder(X)
        # stop gradients from the classifier
        logits = self.classifier(feat.detach())
        return {"logits": logits, "feat": feat}

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        logits = self(X)["logits"]
        loss = F.cross_entropy(logits, target)

        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outs):
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

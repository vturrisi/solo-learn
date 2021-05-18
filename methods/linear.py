import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.lars import LARSWrapper
from utils.metrics import accuracy_at_k, weighted_mean


class LinearModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()

        self.args = args
        self.model = model
        # reset classifier
        self.model.classifier = nn.Linear(self.model.features_size, args.n_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        args = self.args

        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{args.optimizer} not in (sgd, adam)")

        optimizer = optimizer(
            self.model.classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            **args.extra_optimizer_args,
        )
        if args.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=args.exclude_bias_n_norm)

        # select scheduler
        if args.scheduler == "none":
            return optimizer
        else:
            if args.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(optimizer, 10, args.epochs)
            if args.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, args.epochs)
            elif args.scheduler == "reduce":
                scheduler = ReduceLROnPlateau(optimizer)
            elif args.scheduler == "step":
                scheduler = MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
            elif args.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, args.weight_decay)
            return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # set encoder to eval mode and classifier to train mode
        self.model.encoder.eval()
        self.model.classifier.train()

        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def shared_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        output = self(X)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        _, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = weighted_mean(outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outputs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, prog_bar=True, sync_dist=True)

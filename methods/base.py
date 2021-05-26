from abc import abstractmethod
import os
import sys
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.lars import LARSWrapper
from utils.metrics import accuracy_at_k, weighted_mean


def static_lr(get_lr, param_group_indexes, lrs_to_replace):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        assert args.encoder in ["resnet18", "resnet50"]
        from torchvision.models import resnet18, resnet50

        self.base_model = {"resnet18": resnet18, "resnet50": resnet50}[args.encoder]

        # initialize encoder
        self.encoder = self.base_model(zero_init_residual=args.zero_init_residual)
        self.features_size = self.encoder.inplanes
        # remove fc layer
        self.encoder.fc = nn.Identity()
        if args.cifar:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.encoder.maxpool = nn.Identity()

        self.classifier = nn.Linear(self.features_size, args.n_classes)

    @property
    def base_learnable_modules(self):
        return [
            self.encoder,
            {
                "module": self.classifier,
                "lr": self.args.classifier_lr,
                "weight_decay": 0
            }
        ]

    @property
    @abstractmethod
    def extra_learnable_modules(self):
        pass

    def configure_optimizers(self):
        args = self.args

        # collect learnable parameters
        learnable_params = []
        idxs_no_scheduler = []
        base_learnable_modules = list(self.base_learnable_modules)
        extra_learnable_modules = list(self.extra_learnable_modules)
        learnable_modules = base_learnable_modules + extra_learnable_modules
        for idx, module in enumerate(learnable_modules):
            if isinstance(module, nn.Module) or isinstance(module, nn.Parameter):
                learnable_params.append({"params": module.parameters()})
            elif isinstance(module, dict):
                module["params"] = module.pop("module").parameters()
                learnable_params.append(module)
                if module.pop("static_lr", False):
                    idxs_no_scheduler.append(idx)
            else:
                raise ValueError(f"{type(module)} cannot be parsed")

        print(idxs_no_scheduler)

        # select optimizer
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{args.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            **args.extra_optimizer_args,
        )
        # optionally wrap with lars
        if args.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=args.exclude_bias_n_norm)

        if args.scheduler == "none":
            return optimizer
        else:
            if args.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=args.epochs,
                    warmup_start_lr=0.003,
                )
            elif args.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, args.epochs)
            elif args.scheduler == "step":
                scheduler = MultiStepLR(optimizer, args.lr_decay_steps)
            else:
                raise ValueError(f"{args.scheduler} not in (warmup_cosine, cosine, step)")

            if args.no_lr_scheduler_for_pred_head:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[args.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn
            return [optimizer], [scheduler]

    def forward(self, X, classify_only=True):
        feat = self.encoder(X)
        # stop gradients from the classifier
        y = self.classifier(feat.detach())

        if classify_only:
            return y

        return feat, y

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        output = self(X)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

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
        self.log_dict(log, sync_dist=True)
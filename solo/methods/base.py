from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params


def static_lr(get_lr, param_group_indexes, lrs_to_replace):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        n_classes,
        cifar,
        zero_init_residual,
        max_epochs,
        optimizer,
        lars,
        lr,
        weight_decay,
        classifier_lr,
        exclude_bias_n_norm,
        accumulate_grad_batches,
        extra_optimizer_args,
        scheduler,
        min_lr,
        warmup_start_lr,
        multicrop,
        n_crops,
        n_small_crops,
        lr_decay_steps=None,
        **kwargs,
    ):
        super().__init__()

        # back-bone related
        self.cifar = cifar
        self.zero_init_residual = zero_init_residual

        # training related
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.multicrop = multicrop
        self.n_crops = n_crops
        self.n_small_crops = n_small_crops

        # sanity checks on multicrop
        if self.multicrop:
            assert n_small_crops > 0
        else:
            self.n_small_crops = 0

        # all the other parameters
        self.extra_args = kwargs

        # if accumulating gradient then scale lr
        self.lr = self.lr * self.accumulate_grad_batches
        self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
        self.min_lr = self.min_lr * self.accumulate_grad_batches
        self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("base")

        # encoder args
        SUPPORTED_NETWORKS = ["resnet18", "resnet50"]

        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
        parser.add_argument("--zero_init_residual", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)

        return parent_parser

    @property
    def learnable_params(self):
        return [
            {"params": self.encoder.parameters()},
            {"params": self.classifier.parameters(), "lr": self.classifier_lr, "weight_decay": 0},
        ]

    def configure_optimizers(self):
        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
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
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
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

    def forward(self, *args, **kwargs):
        return self._base_forward(*args, **kwargs)

    def _base_forward(self, X, detach_feats=True):
        feats = self.encoder(X)
        logits = self.classifier(feats.detach() if detach_feats else feats)
        return logits, feats

    def _shared_step(self, X, targets):
        logits, feats = self._base_forward(X)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {
            "loss": loss,
            "logits": logits,
            "feats": feats,
            "acc1": acc1,
            "acc5": acc5,
        }

    def training_step(self, batch, batch_idx):
        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.n_crops + self.n_small_crops

        outs = [self._shared_step(x, targets) for x in X[: self.n_crops]]

        # collect data
        logits = [out["logits"] for out in outs]
        feats = [out["feats"] for out in outs]

        # loss and stats
        loss = sum(out["loss"] for out in outs) / self.n_crops
        acc1 = sum(out["acc1"] for out in outs) / self.n_crops
        acc5 = sum(out["acc5"] for out in outs) / self.n_crops

        if self.multicrop:
            feats.append([self.encoder(x) for x in X[-self.n_small_crops :]])

        metrics = {
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_class_loss": loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return {"loss": loss, "feats": feats, "logits": logits}

    def validation_step(self, batch, batch_idx):
        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step(X, targets)

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs):
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)


class BaseMomentumModel(BaseModel):
    def __init__(self, base_tau_momentum, final_tau_momentum, **kwargs):
        super().__init__(**kwargs)

        # momentum encoder
        self.momentum_encoder = self.base_model(zero_init_residual=self.zero_init_residual)
        self.momentum_encoder.fc = nn.Identity()
        if self.cifar:
            self.momentum_encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # momentum classifier
        self.momenutm_classifier = nn.Linear(self.features_size, self.n_classes)

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self):
        momentum_learnable_parameters = [
            {
                "params": self.momentum_classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            }
        ]
        return super().learnable_parameters + momentum_learnable_parameters

    @property
    def momentum_pairs(self):
        return [(self.encoder, self.momentum_encoder)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(BaseMomentumModel, BaseMomentumModel).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)

        return parent_parser

    def on_train_start(self):
        self.last_step = 0

    def forward_momentum(self, X, targets):
        with torch.no_grad():
            feats = self.momentum_encoder(X)
        logits = self.momentum_classifier(feats)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {
            "loss": loss,
            "logits": logits,
            "feats": feats,
            "acc1": acc1,
            "acc5": acc5,
        }

    def training_step(self, batch, batch_idx):
        parent_outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.n_crops]

        outs = [self.forward_momentum(x) for x in X]

        # collect data
        logits = [out["logits"] for out in outs]
        feats = [out["feats"] for out in outs]

        # momentum loss and stats
        loss = sum(out["loss"] for out in outs) / self.n_crops
        acc1 = sum(out["acc1"] for out in outs) / self.n_crops
        acc5 = sum(out["acc5"] for out in outs) / self.n_crops

        metrics = {
            "train_momentum_acc1": acc1,
            "train_momentum_acc5": acc5,
            "train_momentum_class_loss": loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        parent_outs["loss"] += loss
        parent_outs["feats_momentum"] = feats
        parent_outs["logits_momentum"] = logits
        return parent_outs

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.trainer.global_step > self.last_step:
            # update momentum encoder and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

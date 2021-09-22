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

from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from solo.utils.knn import WeightedKNNClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseMethod(pl.LightningModule):

    _SUPPORTED_ENCODERS = [
        "resnet18",
        "resnet50",
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
        "swin_tiny",
        "swin_small",
        "swin_base",
        "swin_large",
    ]

    def __init__(
        self,
        encoder: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: int,
        extra_optimizer_args: Dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        multicrop: bool,
        num_crops: int,
        num_small_crops: int,
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        disable_knn_eval: bool = True,
        knn_k: int = 20,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            encoder (str): architecture of the base encoder.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet encoder.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (int): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            multicrop (bool): flag indicating if multi-resolution crop is being used.
            num_crops (int): number of big crops
            num_small_crops (int): number of small crops (will be set to 0 if multicrop is False).
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            disable_knn_eval (bool): disables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet encoder
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        .. note::
            If multicrop is activated, the number of small crops must be greater than zero. When
            multicrop is deactivated (default) the number of small crops is ignored.

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
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
        self.warmup_epochs = warmup_epochs
        self.multicrop = multicrop
        self.num_crops = num_crops
        self.num_small_crops = num_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars
        self.disable_knn_eval = disable_knn_eval
        self.knn_k = knn_k

        # sanity checks on multicrop
        if self.multicrop:
            assert num_small_crops > 0
        else:
            self.num_small_crops = 0

        # all the other parameters
        self.extra_args = kwargs

        # if accumulating gradient then scale lr
        self.lr = self.lr * self.accumulate_grad_batches
        self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
        self.min_lr = self.min_lr * self.accumulate_grad_batches
        self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert encoder in BaseMethod._SUPPORTED_ENCODERS
        from solo.utils.backbones import (
            swin_base,
            swin_large,
            swin_small,
            swin_tiny,
            vit_base,
            vit_large,
            vit_small,
            vit_tiny,
        )
        from torchvision.models import resnet18, resnet50

        self.base_model = {
            "resnet18": resnet18,
            "resnet50": resnet50,
            "vit_tiny": vit_tiny,
            "vit_small": vit_small,
            "vit_base": vit_base,
            "vit_large": vit_large,
            "swin_tiny": swin_tiny,
            "swin_small": swin_small,
            "swin_base": swin_base,
            "swin_large": swin_large,
        }[encoder]

        self.encoder_name = encoder

        # initialize encoder
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.encoder_name and cifar:
            kwargs["window_size"] = 4

        self.encoder = self.base_model(**kwargs)
        if "resnet" in self.encoder_name:
            self.features_dim = self.encoder.inplanes
            # remove fc layer
            self.encoder.fc = nn.Identity()
            if cifar:
                self.encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.encoder.maxpool = nn.Identity()
        else:
            self.features_dim = self.encoder.num_features

        self.classifier = nn.Linear(self.features_dim, num_classes)

        if not self.disable_knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # encoder args
        SUPPORTED_ENCODERS = BaseMethod._SUPPORTED_ENCODERS

        parser.add_argument("--encoder", choices=SUPPORTED_ENCODERS, type=str)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

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
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
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
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # DALI only
        # uses sample indexes as labels and then gets the labels from a lookup table
        # this may use more CPU memory, so just use when needed.
        parser.add_argument("--encode_indexes_into_labels", action="store_true")

        # online knn eval
        parser.add_argument("--disable_knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
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

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""

        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        feats = self.encoder(X)
        logits = self.classifier(feats.detach())
        return {"logits": logits, "feats": feats}

    def _shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format
            targets (torch.Tensor): batch of labels for X

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5
        """

        out = self.base_forward(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits
        """

        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops + self.num_small_crops

        outs = [self._shared_step(x, targets) for x in X[: self.num_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            outs["feats"].extend([self.encoder(x) for x in X[self.num_crops :]])

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_crops

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if not self.disable_knn_eval:
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_crops]).detach(),
                train_targets=targets.repeat(self.num_crops),
            )

        return outs

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y]
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step(X, targets)

        if not self.disable_knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets)

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if not self.disable_knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)


class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        base_tau_momentum: float,
        final_tau_momentum: float,
        momentum_classifier: bool,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum encoder. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum encoder and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                encoder.
        """

        super().__init__(**kwargs)

        # momentum encoder
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.encoder_name and cifar:
            kwargs["window_size"] = 4

        self.momentum_encoder = self.base_model(**kwargs)
        if "resnet" in self.encoder_name:
            self.features_dim = self.momentum_encoder.inplanes
            # remove fc layer
            self.momentum_encoder.fc = nn.Identity()
            if cifar:
                self.momentum_encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_encoder.maxpool = nn.Identity()
        else:
            self.features_dim = self.momentum_encoder.num_features

        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.encoder, self.momentum_encoder)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumMethod, BaseMomentumMethod).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def base_momentum_forward(self, X: torch.Tensor) -> Dict:
        """Momentum forward that allows children classes to override how the momentum encoder is used.
        Args:
            X (torch.Tensor): batch of images in tensor format.
        Returns:
            Dict: dict of logits and features.
        """

        feats = self.momentum_encoder(X)
        return {"feats": feats}

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum encoder and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum encoder / classifier.
        """

        out = self.base_momentum_forward(X)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)

            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update({"logits": logits, "loss": loss, "acc1": acc1, "acc5": acc5})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum encoder and classifier, such as forwarding the crops in the momentum encoder
        and classifier, and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum encoder and the classification
                loss and logits of the momentum classifier.
        """

        outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = sum(momentum_outs["momentum_loss"]) / self.num_crops
            momentum_outs["momentum_acc1"] = sum(momentum_outs["momentum_acc1"]) / self.num_crops
            momentum_outs["momentum_acc5"] = sum(momentum_outs["momentum_acc5"]) / self.num_crops

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        return {**outs, **momentum_outs}

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """

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

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum encoder and classifier, such as forwarding a batch of images in the momentum
        encoder and classifier and computing statistics.

        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx)

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifier is not None:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"],
                "momentum_val_acc1": out["acc1"],
                "momentum_val_acc5": out["acc5"],
            }

        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum encoder / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            momentum_outs = [out[1] for out in outs]

            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(momentum_outs, "momentum_val_acc1", "batch_size")
            val_acc5 = weighted_mean(momentum_outs, "momentum_val_acc5", "batch_size")

            log = {
                "momentum_val_loss": val_loss,
                "momentum_val_acc1": val_acc1,
                "momentum_val_acc5": val_acc5,
            }
            self.log_dict(log, sync_dist=True)

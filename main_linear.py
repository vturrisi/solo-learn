# Copyright 2022 solo-learn development team.

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

import logging
import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.setup import parse_args_linear
from solo.data.classification_dataloader import prepare_data
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True


def main():
    args = parse_args_linear()

    assert args.backbone in BaseMethod._BACKBONES
    backbone_model = BaseMethod._BACKBONES[args.backbone]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    if "vit" in args.backbone:
        kwargs["drop_path_rate"] = args.drop_path
        kwargs["global_pool"] = args.global_pool

    method = args.pretrain_method
    backbone = backbone_model(method=method, **kwargs)
    if args.backbone.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            logging.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)
    logging.info(f"Loaded {ckpt_path}")

    # check if mixup or cutmix is enabled
    mixup_func = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )
        # smoothing is handled with mixup label transform
        loss_func = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    del args.backbone
    model = LinearModel(backbone, loss_func=loss_func, mixup_func=mixup_func, **args.__dict__)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not args.no_channel_last:
        model = model.to(memory_format=torch.channels_last)

    if args.data_format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = args.data_format

    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=val_data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=args.auto_augment,
    )

    if args.data_format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        assert not args.auto_augment, "Auto augmentation is not supported with Dali."

        dali_datamodule = ClassificationDALIDataModule(
            dataset=args.dataset,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
        )

        # use normal torchvision dataloader for validation to save memory
        dali_datamodule.val_dataloader = lambda: val_loader

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, "linear"),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    # fix for incompatibility with nvidia-dali and pytorch lightning
    # with dali 1.15 (this will be fixed on 1.16)
    # https://github.com/Lightning-AI/lightning/issues/12956
    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    if args.data_format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

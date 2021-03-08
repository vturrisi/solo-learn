import argparse
import json
import os
import sys

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.base import Model
from models.linear import LinearModel
from models.dali import DaliLinearModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.epoch_checkpointer import EpochCheckpointer
from utils.classification_dataloader import prepare_data


def parse_args():
    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
    ]
    SUPPORTED_NETWORKS = ["resnet18", "resnet50"]

    SUPPORTED_OPTIMIZERS = ["sgd", "adam"]
    SUPPORTED_SCHEDULERS = [
        "reduce",
        "cosine",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS, type=str)
    parser.add_argument("encoder", choices=SUPPORTED_NETWORKS, type=str)

    # optimizer
    parser.add_argument(
        "-optimizer", "--optimizer", default="sgd", choices=SUPPORTED_OPTIMIZERS, type=str,
    )
    # scheduler
    parser.add_argument(
        "-scheduler", "--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce",
    )
    parser.add_argument(
        "-lr_decay_steps", "--lr_decay_steps", default=[200, 300, 350], type=int, nargs="+",
    )

    # general settings
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--no_projection_bn", action="store_true")

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, nargs="+")

    # pretrained model path
    parser.add_argument("--pretrained_feature_extractor")

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")

    # dataset path
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    # this only works for imagenet, but you prob won't need to any other of the "normal" datasets
    parser.add_argument("--dali", action="store_true")

    args = parser.parse_args()

    if args.dataset == "cifar10":
        args.n_classes = 10
    elif args.dataset == "cifar100":
        args.n_classes = 100
    elif args.dataset == "stl10":
        args.n_classes = 10
    elif args.dataset == "imagenet":
        args.n_classes = 1000
    elif args.dataset == "imagenet100":
        args.n_classes = 100

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    args.projection_bn = not args.no_projection_bn

    args.n_projection_heads = 1

    return args


def main():
    args = parse_args()

    model_args_path = os.path.join(args.pretrained_feature_extractor, "args.json")
    model_args_dict = dict(**json.load(open(model_args_path)))
    model_args = argparse.Namespace(**model_args_dict)

    model = Model(model_args)
    if (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    ):
        ckpt_path = args.pretrained_feature_extractor
    else:
        # load pretrained model (i.e., feature extractor)
        checkpoints = [
            f
            for f in os.listdir(args.pretrained_feature_extractor)
            if f.endswith(".ckpt") or f.endswith(".pth") or f.endswith(".pt")
        ]
        checkpoints.sort(key=lambda ckpt: int(ckpt[:-5].split("ep=")[1]), reverse=True)
        ckpt_path = os.path.join(args.pretrained_feature_extractor, checkpoints[0])

    state = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state, strict=False)

    if args.dali:
        model = DaliLinearModel(model, args)
    else:
        model = LinearModel(model, args)

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_folder=args.data_folder,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # wandb logging
    wandb_logger = WandbLogger(name=args.name, project=args.project)
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # epoch checkpointer
    checkpointer = EpochCheckpointer(args, frequency=25)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger,
        distributed_backend="ddp",
        precision=16,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=[lr_monitor, checkpointer],
        num_sanity_val_steps=0 if args.dali else 2,
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

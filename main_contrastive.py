import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.base import Model
from models.simclr import SimCLR
from models.dali import DaliSimCLR

from models.deepcluster import (
    resnet18 as deepcluster_resnet18,
    resnet50 as deepcluster_resnet50,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.contrastive_dataloader import prepare_data, prepare_data_multicrop
from utils.epoch_checkpointer import EpochCheckpointer
from utils.info_nce import info_nce
from utils.normal_dataloader import prepare_data as prepare_data_normal


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
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--encoding_size", type=int, default=128)
    parser.add_argument("--hidden_mlp", type=int, default=2048)
    parser.add_argument("--lars", type=bool, default=True)
    parser.add_argument("--no_projection_bn", action="store_true")

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--precision", type=int, default=16)

    # extra dataloader settings
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--with_index", action="store_true")
    parser.add_argument("--pseudo_labels_path", default=None)
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--last_batch_fill", action="store_true")

    # dataset path
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)

    # extra simclr settings
    parser.add_argument("--supervised", action="store_true")

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")

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

    args.lr = args.lr * args.batch_size * len(args.gpus) / 256

    # extra deepcluster settings
    args.deepcluster_hidden_mlp = 2048

    args.projection_bn = not args.no_projection_bn

    return args


def main():
    seed_everything(5)

    args = parse_args()

    if args.dali:
        model = DaliSimCLR(args)
    else:
        model = SimCLR(args)

    # contrastive dataloader
    if not args.dali:
        if args.multicrop:
            train_loader, _ = prepare_data_multicrop(
                args.dataset,
                data_folder=args.data_folder,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                nmb_crops=[args.n_crops, args.n_small_crops]
                if args.deepcluster_simclr
                else None,
                consensus=False,
                with_index=args.with_index,
                pseudo_labels_path=args.pseudo_labels_path,
            )
        else:
            train_loader, _ = prepare_data(
                args.dataset,
                data_folder=args.data_folder,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                n_augs=2,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                with_index=args.with_index,
                pseudo_labels_path=args.pseudo_labels_path,
            )

    # normal dataloader
    _, val_loader = prepare_data_normal(
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
        precision=args.precision,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=[lr_monitor, checkpointer],
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

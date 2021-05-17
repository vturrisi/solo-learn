import argparse
import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from args.setup import parse_args_linear
from methods.base import Model
from methods.dali import DaliLinearModel
from methods.linear import LinearModel
from utils.classification_dataloader import prepare_data
from utils.epoch_checkpointer import EpochCheckpointer


def main():
    args = parse_args_linear()

    model_args_path = os.path.join(args.pretrained_feature_extractor, "args.json")
    model_args_dict = dict(**json.load(open(model_args_path)))
    model_args = argparse.Namespace(**model_args_dict)

    # compatibility with encoders created before zero_init_residual was added
    if "zero_init_residual" not in model_args:
        model_args.zero_init_residual = False

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

    print(f"loaded {ckpt_path}")
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

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        # lr logging
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # epoch checkpointer
    callbacks.append(EpochCheckpointer(args, frequency=25))

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger if args.wandb else None,
        distributed_backend="ddp",
        precision=16,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
        num_sanity_val_steps=0 if args.dali else 2,
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

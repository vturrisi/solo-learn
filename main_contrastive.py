from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from args.setup import parse_args_contrastive
from methods import METHODS
from methods.dali import ContrastiveABC
from utils.classification_dataloader import prepare_data as prepare_data_classification
from utils.contrastive_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)
from utils.epoch_checkpointer import EpochCheckpointer


def main():
    seed_everything(5)

    args = parse_args_contrastive()

    assert args.method in METHODS.keys(), f"Choose from {METHODS.keys()}"
    MethodClass = METHODS[args.method]

    if args.dali:
        MethodClass = type(f"Dali{MethodClass.__name__}", (MethodClass, ContrastiveABC), {})

    model = MethodClass(args)

    # contrastive dataloader
    if not args.dali:
        # asymmetric augmentations on barlow twins
        if args.asymmetric_augmentations:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.asymmetric_augmentations

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, n_crops=[args.n_crops, args.n_small_crops]
            )
        else:
            transform = prepare_n_crop_transform(transform, n_crops=2)

        train_dataset = prepare_datasets(
            args.dataset,
            data_folder=args.data_folder,
            train_dir=args.train_dir,
            transform=transform,
        )
        train_loader = prepare_dataloaders(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader
    _, val_loader = prepare_data_classification(
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
        precision=args.precision,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

import os
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from solo.args.setup import parse_args_contrastive
from solo.methods import METHODS

try:
    from solo.methods.dali import ContrastiveABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.contrastive_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)


def main():
    seed_everything(5)

    args = parse_args_contrastive()

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        MethodClass = type(f"Dali{MethodClass.__name__}", (MethodClass, ContrastiveABC), {})

    model = MethodClass(**args.__dict__)

    # contrastive dataloader
    if not args.dali:
        # asymmetric augmentations
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
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

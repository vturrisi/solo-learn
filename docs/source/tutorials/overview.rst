An Overview
***********

This tutorial aims to guide the user on how to rebuild ``main_pretrain.py`` to provide a general view of all the components of the library.

Let's first go through how the library is organized.

#. All different methods are located in ``solo/methods``, with a single file per method and some additional files to provide utilities for all methods.

#. The losses used by the methods are contained in ``solo/losses``.

#. Parsing utilities are contained in ``solo/args``.

#. And all utility classes (e.g. data loading, extra features, kmeans, lars and so on) are contained in ``solo/utils``

#. ``main_pretrain.py`` is the main script that trains any of the available methods.

#. ``main_linear.py`` is a script to train linear accuracy on a pretrained backbone.

Now, let's assume that we want to train Barlow Twins on CIFAR10 for 100 epochs.
For this, we won't use the ``main_pretrain.py`` file directly, but we'll build a minimal version of it in order to give a general overview of the library.

We start by importing everything that we will need (we will be relying on Pytorch Lightning to use our already implemented training/validation steps:

.. code-block:: python

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.plugins import DDPPlugin

    from solo.methods import BarlowTwins  # imports the method class
    from solo.utils.checkpointer import Checkpointer

    # some data utilities
    # we need one dataloader to train an online linear classifier
    # (don't worry, the rest of the model has no idea of this classifier, so it doesn't use label info)
    from solo.utils.classification_dataloader import prepare_data as prepare_data_classification

    # and some utilities to perform data loading for the method itself, including augmentation pipelines
    from solo.utils.pretrain_dataloader import (
        prepare_dataloader,
        prepare_datasets,
        prepare_n_crop_transform,
        prepare_transform,
    )


There are tons of parameters that need to be set and, fortunately, ``main_pretrain.py`` takes care of this for us.
We can also call specific methods from the many ``solo`` classes to automatically change an argparse and add new subparsers.
However, for now, we won't rely on this, so let's just define all the needed parameters and create a Barlow Twins object:

.. code-block:: python

    # common parameters for all methods
    # some parameters for extra functionally are missing, but don't mind this for now.
    base_kwargs = {
        "backbone": "resnet18",
        "num_classes": 10,
        "cifar": True,
        "zero_init_residual": True,
        "max_epochs": 100,
        "optimizer": "sgd",
        "lars": True,
        "lr": 0.01,
        "gpus": "0",
        "grad_clip_lars": True,
        "weight_decay": 0.00001,
        "classifier_lr": 0.5,
        "exclude_bias_n_norm_lars": True,
        "accumulate_grad_batches": 1,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "warmup_cosine",
        "min_lr": 0.0,
        "warmup_start_lr": 0.0,
        "warmup_epochs": 10,
        "num_crops_per_aug": [2, 0],
        "num_large_crops": 2,
        "num_small_crops": 0,
        "eta_lars": 0.02,
        "lr_decay_steps": None,
        "dali_device": "gpu",
        "batch_size": 256,
        "num_workers": 4,
        "data_dir": "/data/datasets",
        "train_dir": "cifar10/train",
        "val_dir": "cifar10/val",
        "dataset": "cifar10",
        "name": "barlow-cifar10",
    }

    # barlow specific parameters
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    }

    kwargs = {**base_kwargs, **method_kwargs}

    model = BarlowTwins(**kwargs)


Now, let's create all the necessary data loaders.

.. code-block:: python

    # we first prepare our single transformation pipeline
    transform_kwargs = {
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_prob": 0.0,
        "solarization_prob": 0.0,
    }
    transform = [prepare_transform("cifar10", **transform_kwargs)]

    # then, we wrap the pipepline using this utility function
    # to make it produce an arbitrary number of crops
    transform = prepare_n_crop_transform(transform, num_crops_per_aug=[2])

    # finally, we produce the Dataset/Dataloader classes
    train_dataset = prepare_datasets(
        "cifar10",
        transform,
        data_dir="./",
        train_dir=None,
        no_labels=False,
    )
    train_loader = prepare_dataloader(
        train_dataset, batch_size=base_kwargs["batch_size"], num_workers=base_kwargs["num_workers"]
    )

    # we will also create a validation dataloader to automatically
    # check how well our models is doing in an online fashion.
    _, val_loader = prepare_data_classification(
        "cifar10",
        data_dir="./",
        train_dir=None,
        val_dir=None,
        batch_size=base_kwargs["batch_size"],
        num_workers=base_kwargs["num_workers"],
    )

Now, we just need to define some extra magic for Pytorch Lightning to automatically log some stuff for us and then we can just create our lightning Trainer.

.. code-block:: python

    wandb_logger = WandbLogger(
        name="barlow-cifar10",  # name of the experiment
        project="self-supervised",  # name of the wandb project
        entity=None,
        offline=False,
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)

    callbacks = []

    # automatically log our learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # checkpointer can automatically log your parameters,
    # but we need to wrap it on a Namespace object
    from argparse import Namespace

    args = Namespace(**kwargs)
    # saves the checkout after every epoch
    ckpt = Checkpointer(
        args,
        logdir="checkpoints/barlow",
        frequency=1,
    )
    callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )

    trainer.fit(model, train_loader, val_loader)


And that's it, we basically replicated a small version of ``main_pretrain.py``. Of course, we can accomplish the same thing by simply running the following script:

.. code-block:: bash

    python3 main_pretrain.py \
        --dataset cifar10 \
        --backbone resnet18 \
        --data_dir ./datasets \
        --max_epochs 1000 \
        --gpus 0 \
        --num_workers 4 \
        --precision 16 \
        --optimizer sgd \
        --lars \
        --grad_clip_lars \
        --eta_lars 0.02 \
        --exclude_bias_n_norm_lars \
        --scheduler warmup_cosine \
        --lr 0.3 \
        --weight_decay 1e-4 \
        --batch_size 256 \
        --brightness 0.4 \
        --contrast 0.4 \
        --saturation 0.2 \
        --hue 0.1 \
        --gaussian_prob 0.0 \
        --solarization_prob 0.0 \
        --name barlow-cifar10 \
        --project self-superivsed \
        --wandb \
        --save_checkpoint \
        --method barlow_twins \
        --proj_hidden_dim 2048 \
        --output_dim 2048 \
        --scale_loss 0.1

There are tons of extra options! You can use LARS, use different precisions, optimizers, learning rate schedulers, create asymmetric augmentation pipelines and so on!
We hope that this tutorial gives a general overview of how to use what is already implemented.

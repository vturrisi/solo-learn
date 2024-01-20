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

We start by importing everything that we will need (we will be relying on Pytorch Lightning to use our already implemented training/validation steps):

.. code-block:: python

    import torch
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins import DDPPlugin

    # solo learn uses omega conf and hydra to manage configs files now
    from omegaconf  import DictConfig
    from solo.methods import BarlowTwins  # imports the method class
    from solo.utils.checkpointer import Checkpointer

    # some data utilities
    # we need one dataloader to train an online linear classifier
    # (don't worry, the rest of the model has no idea of this classifier, so it doesn't use label info)
    from solo.data.classification_dataloader import prepare_data as prepare_classification_dataloader

    # and some utilities to perform data loading for the method itself, including augmentation pipelines
    from solo.data.pretrain_dataloader import (
        build_transform_pipeline,
        prepare_dataloader,
        prepare_datasets,
        prepare_n_crop_transform,
    )


There are tons of parameters that need to be set and, fortunately, ``main_pretrain.py`` takes care of this for us.
We can also call specific methods from the many ``solo`` classes to automatically change an argparse and add new subparsers.
However, for now, we won't rely on this, so let's just define all the needed parameters and create a Barlow Twins object:

.. code-block:: python

    # common parameters for all methods
    # some parameters for extra functionally are missing, but don't mind this for now.
    base_kwargs = {
        "name": "barlow_twins-cifar10", # change here for cifar100
        "backbone": {
            "name": "resnet18",
            "kwargs": {}
        },
        "data": {
            "dataset": "cifar10",
            "num_classes": 10,
            "train_path": "./data",  # replace with your own path
            "val_path": "./data", # replace with your own path
            "num_large_crops": 2, # must equal 2 for barlow twins
            "num_small_crops": 0, # must equal 0 for barlow twins
            "num_workers": 4,
        },
        "cifar": True,
        "zero_init_residual": True,
        "max_epochs": 100,
        "optimizer": {
            "name": "lars",
            "lr": 0.01,
            "batch_size": 256,
            "weight_decay": 0.00001,
            "classifier_lr": 0.1       # mandatory

        },
        "scheduler":{
            "name": "warmup_cosine",
            "min_lr": 0.0,
            "warmup_start_lr": 0.0,
            "warmup_epochs": 10,
        },
        "method": "barlow_twins",
        "dali_device": "gpu",
    }

    # barlow specific parameters
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    }

    cfg = DictConfig({**base_kwargs, "method_kwargs": method_kwargs})
    model = BarlowTwins(cfg)


Now, let's create all the necessary data loaders.

.. code-block:: python

    # we first prepare our single transformation pipeline config
    transform_kwargs = {
        "crop_size": 32,
        "num_crops": 1,
        "rrc": {
            "enabled": True,
            "crop_min_scale": 0.08,
            "crop_max_scale": 1.0
        },
        "color_jitter": {
            "prob":  0.8,
            "brightness":  0.4,
            "contrast":  0.4,
            "saturation":  0.2,
            "hue":  0.1,
        },
        # all below need to be specified but are unused
        "grayscale": {"prob": 0.0},
        "gaussian_blur": {"prob": 0.0},
        "solarization": {"prob": 0.0},
        "equalization": {"prob": 0.0},
        "horizontal_flip": {"prob": 0.0},
    }
    aug_cfg = DictConfig(transform_kwargs)
    augs = build_transform_pipeline("cifar10", aug_cfg)


    # then, we wrap the pipeline using this utility function
    # to make it produce an arbitrary number of crops
    transform = prepare_n_crop_transform([augs], num_crops_per_aug=[2])

    # finally, we produce the Dataset/Dataloader classes
    train_dataset = prepare_datasets(
        dataset="cifar10",
        transform=transform,
        train_data_path=base_kwargs["data"]["train_path"],
        no_labels=False,
    )
    train_loader = prepare_dataloader(
        train_dataset=train_dataset,
        batch_size=base_kwargs["optimizer"]["batch_size"],
        num_workers=base_kwargs["data"]["num_workers"]
    )

    # we will also create a validation dataloader to automatically
    # check how well our models is doing in an online fashion.
    _, val_loader = prepare_classification_dataloader(
        dataset=base_kwargs["data"]["dataset"],  # "cifar10"
        train_data_path=base_kwargs["data"]["train_path"],
        val_data_path=base_kwargs["data"]["val_path"],
        batch_size=base_kwargs["optimizer"]["batch_size"],
        num_workers=base_kwargs["data"]["num_workers"],
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

    # saves the checkout after every epoch
    ckpt = Checkpointer(
        cfg,
        logdir="checkpoints/barlow",
        frequency=1,
    )
    callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        cfg,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto", # use whatever is available
        strategy="ddp",     # could change  depending on your setup
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

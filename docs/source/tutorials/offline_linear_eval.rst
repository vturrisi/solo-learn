Offline Linear Eval
*******************

Now that you know how to pretrain a model, let's go through the procedure to perform offline linear evaluation.

As for pretraining, we start by importing the required packages:

.. code-block:: python

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor
    from torchvision.models import resnet18

    from solo.methods.linear import LinearModel  # imports the linear eval class
    from solo.utils.classification_dataloader import prepare_data

There are tons of parameters that need to be set and, fortunately, ``main_linear.py`` takes care of this for us.
If we want to be able to specify the arguments from the command line, we can simply call the function ``parse_args_linear`` in ``solo.args.setup``.
However, in this tutorial, we will simply define all the needed parameters to perform linear evaluation:

.. code-block:: python

    # basic parameters for offline linear evaluation
    # some parameters for extra functionally are missing, but don't mind this for now.
    kwargs = {
        "num_classes": 10,
        "cifar": True,
        "max_epochs": 100,
        "optimizer": "sgd",
        "precision": 16,
        "lars": False,
        "lr": 0.1,
        "exclude_bias_n_norm_lars": False,
        "gpus": "0",
        "weight_decay": 0,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "step",
        "lr_decay_steps": [60, 80],
        "batch_size": 128,
        "num_workers": 4,
        "pretrained_feature_extractor": "path/to/pretrained/feature/extractor"
    }

Apart from the hyperparameters, we also need to load the pretrained model:

.. code-block:: python

    # create the backbone network
    # the first convolutional and maxpooling layers of the ResNet backbone
    # are adjusted to handle lower resolution images (32x32 instead of 224x224).
    backbone = resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()

    # load pretrained feature extractor
    state = torch.load(kwargs["pretrained_feature_extractor"])["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    model = LinearModel(backbone, **kwargs)

Now, let's create the data loaders. Unlike when we are doing pretraining, this time we will not use multiple augmentations:

.. code-block:: python

    train_loader, val_loader = prepare_data(
        "cifar10",
        data_dir="./",
        train_dir=None,
        val_dir=None,
        batch_size=base_kwargs["batch_size"],
        num_workers=base_kwargs["num_workers"],
    )

Lastly, we just need to define some extra utilities for Pytorch Lightning to automatically log some stuff for us and then we can just create our lightning Trainer:

.. code-block:: python

    wandb_logger = WandbLogger(
        name="linear-cifar10",  # name of the experiment
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
    # but we need to wrap them in a Namespace object
    from argparse import Namespace
    args = Namespace(**kwargs)
    # saves the checkout after every epoch
    ckpt = Checkpointer(
        args,
        logdir="checkpoints/linear",
        frequency=1,
    )
    callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
    )

    trainer.fit(model, train_loader, val_loader)

And that's it, we basically replicated a small version of ``main_linear.py``. Of course, we can accomplish the same thing by simply running the following script:

.. code-block:: bash

    python3 ../../main_linear.py \
        --dataset cifar10 \
        --backbone resnet18 \
        --data_dir ./ \
        --max_epochs 100 \
        --gpus 0 \
        --sync_batchnorm \
        --precision 16 \
        --optimizer sgd \
        --scheduler step \
        --lr 0.1 \
        --lr_decay_steps 60 80 \
        --weight_decay 0 \
        --batch_size 128 \
        --num_workers 4 \
        --name general-linear-eval \
        --pretrained_feature_extractor path/to/pretrained/feature/extractor \
        --project self-supervised \
        --wandb

Now you are fully able to use solo-learn and you can make your research ideas become reality!

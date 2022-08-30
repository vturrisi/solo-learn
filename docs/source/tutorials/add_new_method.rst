Adding New Methods
******************

In order to use `solo-learn` to the fullest, you need to be able to add new methods to the existing library. Luckily, the repository has a modular design that enables seamless extension. In this tutorial, we will go through a simple procedure to add a new method.

Let's suppose we wanted to implement `NNSiam <https://arxiv.org/abs/2104.14548/>`_. The fist thing to do is to navigate to `solo/methods` and create a python file for our new method (e.g.: `nnsiam.py`). This file will contain all the code that is specific to NNSiam. The basic functionalities that are shared with all self-supervised methods will not be re-implemented, but inherited from the base class. For non-momentum-based methods, the base class that should be used as a parent is the `BaseMethod`, which can be found in `solo/methods/base.py`. Here you can see how it looks like in the code:

.. code-block:: python

    import argparse
    from typing import Any, Dict, List, Sequence, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from solo.losses.simsiam import simsiam_loss_func
    from solo.methods.base import BaseMethod
    from solo.utils.misc import gather


    class NNSiam(BaseMethod):
        def __init__(
            self,
            proj_output_dim: int,
            proj_hidden_dim: int,
            pred_hidden_dim: int,
            queue_size: int,
            **kwargs,
        ):
            """Implements NNSiam (https://arxiv.org/abs/2104.14548).

            Args:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
                queue_size (int): number of samples to keep in the queue.
            """

            super().__init__(**kwargs)

            self.queue_size = queue_size

            # projector
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
                nn.BatchNorm1d(proj_output_dim, affine=False),
            )
            self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

            # predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, proj_output_dim),
            )

            # queue
            self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
            self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.queue = F.normalize(self.queue, dim=1)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

As you can see, `NNSiam` is a subclass of `BaseMethod` and only contains a projector, a predictor and a queue. Note that the backbone is handled in the base class.

Now we need to make sure that these new modules that we added will be trained along with the other parameters. To achieve this, it is enough to override `learnable_params` a property of the base class that holds the learnable parameters (i.e. the parameters that will be put in the optimizer).

.. code-block:: python

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

Note that the queue is not included in the learnable parameters because gradient is not backpropagated through it.

For convenience, we can also define a forward function in order for our model to be used for inference from the outside:

.. code-block:: python

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

However, to use the model in inference we first need to train the networks. Training is performed using the `PyTorchLightning Trainer <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html>`_, which requires us to implement the `training_step` method:

.. code-block:: python

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for NNSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss
        """

        targets = batch[-1]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # find nn
        idx1, nn1 = self.find_nn(z1)
        _, nn2 = self.find_nn(z2)

        # ------- negative cosine similarity loss -------
        neg_cos_sim = simsiam_loss_func(p1, nn2) / 2 + simsiam_loss_func(p2, nn1) / 2

        # compute nn accuracy
        b = targets.size(0)
        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, targets)

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_nn_acc": nn_acc,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss

As you can see, the training step is very similar to SimSiam. The only difference is that here the negative cosine similarity loss takes the neighbor of the other view as target. Neighbors are mined using the method `find_nn` and the queue is updated using `dequeue_and_enqueue`. You can find both implementations in `solo/methods/nnclr.py`.

Now that it is all ready, we just need to make the method available for the library to use it by simply adding it to `solo/methods/__init__.py`:

.. code-block:: python

    # other imports ...
    from solo.methods.nnclr import NNCLR
    from solo.methods.nnsiam import NNSiam # <--- add this import
    from solo.methods.ressl import ReSSL
    # ... other imports

    METHODS = {
        # other methods ...
        "nnclr": NNCLR,
        "nnsiam": NNSiam, # <--- add this key value pair
        "ressl": ReSSL,
        # ... other methods
    }

    __all__ = [
        # other methods ...
        "NNCLR",
        "NNSiam", # <--- add this string
        "ReSSL",
        # ... other methods
    ]

And that is it, you are good to go. You can now run your implementation of NNSiam using the following command:

.. code-block:: bash

    python3 ../../../main_pretrain.py \
        --dataset DATASET \
        --backbone resnet18 \
        --data_dir DATA_DIR \
        --train_dir TRAIN_DIR \
        --val_dir VAL_DIR \
        --max_epochs 1000 \
        --gpus 0 \
        --precision 16 \
        --optimizer sgd \
        --scheduler warmup_cosine \
        --lr 0.5 \
        --classifier_lr 0.1 \
        --weight_decay 1e-5 \
        --batch_size 256 \
        --num_workers 4 \
        --brightness 0.4 \
        --contrast 0.4 \
        --saturation 0.4 \
        --hue 0.1 \
        --gaussian_prob 0.0 0.0 \
        --num_crops_per_aug 1 1 \
        --zero_init_residual \
        --name nnsiam-DATASET \
        --project solo-learn \
        --entity YOUR_ENTITY \
        --wandb \
        --save_checkpoint \
        --method nnsiam \
        --proj_hidden_dim 2048 \
        --pred_hidden_dim 4096 \
        --proj_output_dim 2048

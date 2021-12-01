Adding New Methods (Momentum Version)
*************************************

As of now, you should be familiar with how to implement new methods in `solo-learn`. If not, please read this tutorial: :doc:`Adding New Methods <add_new_method>`. This tutorial will help you creating methods that use a momentum backbone.
Let's now suppose we wanted to implement NNBYOL (similar to `NNSiam <https://arxiv.org/abs/2104.14548/>`_ but with momentum backbone). As always, the fist thing to do is to navigate to `solo/methods` and create a python file for our new method (e.g.: `nnbyol.py`):

.. code-block:: python

    import argparse
    from typing import Any, Dict, List, Sequence, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from solo.losses.byol import byol_loss_func
    from solo.methods.base import BaseMomentumMethod
    from solo.utils.momentum import initialize_momentum_params
    from solo.utils.misc import gather


    class NNBYOL(BaseMomentumMethod):
        def __init__(
            self,
            proj_output_dim: int,
            proj_hidden_dim: int,
            pred_hidden_dim: int,
            queue_size: int,
            **kwargs,
        ):
            """Implements NNBYOL (https://arxiv.org/abs/2104.14548).

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
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )

            # momentum projector
            self.momentum_projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
            initialize_momentum_params(self.projector, self.momentum_projector)

            # predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_output_dim, pred_hidden_dim),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, proj_output_dim),
            )

            # queue
            self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
            self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.queue = F.normalize(self.queue, dim=1)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        @property
        def learnable_params(self) -> List[dict]:
            """Adds projector and predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.projector.parameters()},
                {"params": self.predictor.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

Note that here we are inheriting from `BaseMomentumMethod` which already implements most of the complexity for momentum-based models. Apart from this, and similarly to `NNSiam`, `NNBYOL` has a projector, a predictor and a queue. However, NNBYOL also has a momentum backbone and a momentum projector that need to be updated at every step. The library already implements this behavior for the momentum backbone. To achieve the same for the momentum projector, the only thing that you need to do is overriding the `momentum_pairs` property of the parent:

.. code-block:: python

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

You can just use the momentum encoder in your training step:

.. code-block:: python

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        targets = batch[-1]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # forward momentum backbone
        with torch.no_grad():
            z1_momentum = self.momentum_projector(momentum_feats1)
            z2_momentum = self.momentum_projector(momentum_feats2)

        z1_momentum = F.normalize(z1_momentum, dim=-1)
        z2_momentum = F.normalize(z2_momentum, dim=-1)

        # find nn
        idx1, nn1_momentum = self.find_nn(z1_momentum)
        _, nn2_momentum = self.find_nn(z2_momentum)

        # ------- negative cosine similarity loss -------
        neg_cos_sim = byol_loss_func(p1, nn2_momentum) + byol_loss_func(p2, nn1_momentum)

        # compute nn accuracy
        b = targets.size(0)
        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1_momentum, targets)

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

For the rest of the code for NNBYOL, please check out the implementation in `solo/methods/nnbyol.py`.

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.ressl import ressl_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.gather_layer import gather
from solo.utils.momentum import initialize_momentum_params


class ReSSL(BaseMomentumModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        temperature_q: float,
        temperature_k: float,
        queue_size: int,
        **kwargs,
    ):
        """Implements ReSSL (https://arxiv.org/abs/2107.09282v1).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
            temperature_q (float): temperature for the contrastive augmentations.
            temperature_k (float): temperature for the weak augmentation.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        self.temperature_q = temperature_q
        self.temperature_k = temperature_k
        self.queue_size = queue_size

        # queue
        self.register_buffer("queue", torch.randn(self.queue_size, output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(ReSSL, ReSSL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ressl")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        # parameters
        parser.add_argument("--temperature_q", type=float, default=0.1)
        parser.add_argument("--temperature_k", type=float, default=0.04)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def dequeue_and_enqueue(self, k: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            z (torch.Tensor): batch of projected features.
        """

        k = gather(k)

        batch_size = k.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = k
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online encoder (encoder, projector and predictor).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X, *args, **kwargs)
        q = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "q": q}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, _ = out["feats"]
        _, momentum_feats2 = out["momentum_feats"]

        q = self.projector(feats1)

        # forward momentum encoder
        with torch.no_grad():
            k = self.momentum_projector(momentum_feats2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # ------- contrastive loss -------
        queue = self.queue.clone().detach()
        ressl_loss = ressl_loss_func(q, k, queue, self.temperature_q, self.temperature_k)

        self.log("ressl_loss", ressl_loss, on_epoch=True, sync_dist=True)

        # dequeue and enqueue
        self.dequeue_and_enqueue(k)

        return ressl_loss + class_loss

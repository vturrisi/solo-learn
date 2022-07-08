# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.wmse import wmse_loss_func
from solo.methods.base import BaseMethod
from solo.utils.whitening import Whitening2d


class WMSE(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        whitening_iters: int,
        whitening_size: int,
        whitening_eps: float,
        **kwargs
    ):
        """Implements W-MSE (https://arxiv.org/abs/2007.06346)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            whitening_iters (int): number of times to perform whitening.
            whitening_size (int): size of the batch slice for whitening.
            whitening_eps (float): epsilon for numerical stability in whitening.
        """

        super().__init__(**kwargs)

        self.whitening_iters = whitening_iters
        self.whitening_size = whitening_size

        assert self.whitening_size <= self.batch_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.whitening = Whitening2d(proj_output_dim, eps=whitening_eps)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(WMSE, WMSE).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=1024)

        # wmse
        parser.add_argument("--whitening_iters", type=int, default=1)
        parser.add_argument("--whitening_size", type=int, default=256)
        parser.add_argument("--whitening_eps", type=float, default=0)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for W-MSE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: total loss composed of W-MSE loss and classification loss
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        v = torch.cat(out["z"])

        # ------- wmse loss -------
        bs = self.batch_size
        num_losses, wmse_loss = 0, 0
        for _ in range(self.whitening_iters):
            z = torch.empty_like(v)
            perm = torch.randperm(bs).view(-1, self.whitening_size)
            for idx in perm:
                for i in range(self.num_large_crops):
                    z[idx + i * bs] = self.whitening(v[idx + i * bs]).type_as(z)
            for i in range(self.num_large_crops - 1):
                for j in range(i + 1, self.num_large_crops):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    wmse_loss += wmse_loss_func(x0, x1)
                    num_losses += 1
        wmse_loss /= num_losses

        self.log("train_wmse_loss", wmse_loss, on_epoch=True, sync_dist=True)

        return wmse_loss + class_loss

# Copyright 2023 solo-learn development team.

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

import omegaconf
import torch
import torch.nn as nn
from solo.losses.wmse import wmse_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.whitening import Whitening2d


class WMSE(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements W-MSE (https://arxiv.org/abs/2007.06346)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                whitening_iters (int): number of times to perform whitening.
                whitening_size (int): size of the batch slice for whitening.
                whitening_eps (float): epsilon for numerical stability in whitening.
        """

        super().__init__(cfg)

        self.whitening_iters: int = cfg.method_kwargs.whitening_iters
        self.whitening_size: int = cfg.method_kwargs.whitening_size

        assert self.whitening_size <= self.batch_size

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        whitening_eps: float = cfg.method_kwargs.whitening_eps

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.whitening = Whitening2d(proj_output_dim, eps=whitening_eps)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(WMSE, WMSE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.whitening_iters = omegaconf_select(
            cfg,
            "method_kwargs.whitening_iters",
            1,
        )
        cfg.method_kwargs.whitening_size = omegaconf_select(
            cfg,
            "method_kwargs.whitening_size",
            256,
        )
        cfg.method_kwargs.whitening_eps = omegaconf_select(cfg, "method_kwargs.whitening_eps", 0.0)

        return cfg

    @property
    def learnable_params(self) -> List[Dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
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

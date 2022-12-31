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
from solo.losses.vibcreg import vibcreg_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.whitening import IterNorm


class VIbCReg(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements VIbCReg (https://arxiv.org/abs/2109.00783)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                sim_loss_weight (float): weight of the invariance term.
                var_loss_weight (float): weight of the variance term.
                cov_loss_weight (float): weight of the covariance term.
                iternorm (bool): If true, an IterNorm layer will be appended to the projector.
        """

        super().__init__(cfg)

        self.sim_loss_weight: float = cfg.method_kwargs.sim_loss_weight
        self.var_loss_weight: float = cfg.method_kwargs.var_loss_weight
        self.cov_loss_weight: float = cfg.method_kwargs.cov_loss_weight

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        iternorm: bool = cfg.method_kwargs.iternorm

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            IterNorm(proj_output_dim, num_groups=64, T=5, dim=2) if iternorm else nn.Identity(),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(VIbCReg, VIbCReg).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.sim_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.sim_loss_weight",
            25.0,
        )
        cfg.method_kwargs.var_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.var_loss_weight",
            25.0,
        )
        cfg.method_kwargs.cov_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.cov_loss_weight",
            200.0,
        )
        cfg.method_kwargs.iternorm = omegaconf_select(cfg, "method_kwargs.iternorm", False)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
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
        """Training step for VIbCReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VIbCReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- vibcreg loss -------
        vibcreg_loss = vibcreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train_vibcreg_loss", vibcreg_loss, on_epoch=True, sync_dist=True)

        return vibcreg_loss + class_loss

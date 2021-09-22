# Copyright 2021 solo-learn development team.

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

import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from einops import repeat
from solo.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from solo.methods.base import BaseMethod


class SimCLR(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        supervised: bool = False,
        **kwargs
    ):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            supervised (bool): whether or not to use supervised contrastive loss. Defaults to False.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.supervised = supervised

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR, SimCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # supervised-simclr
        parser.add_argument("--supervised", action="store_true")
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    @torch.no_grad()
    def gen_extra_positives_gt(self, Y: torch.Tensor) -> torch.Tensor:
        """Generates extra positives for supervised contrastive learning.

        Args:
            Y (torch.Tensor): labels of the samples of the batch.

        Returns:
            torch.Tensor: matrix with extra positives generated using the labels.
        """

        if self.multicrop:
            n_augs = self.num_crops + self.num_small_crops
        else:
            n_augs = 2
        labels_matrix = repeat(Y, "b -> c (d b)", c=n_augs * Y.size(0), d=n_augs)
        labels_matrix = (labels_matrix == labels_matrix.t()).fill_diagonal_(False)
        return labels_matrix

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR and supervised SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes, *_, target = batch

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        if self.multicrop:
            n_augs = self.num_crops + self.num_small_crops

            feats = out["feats"]

            z = torch.cat([self.projector(f) for f in feats])

            # ------- contrastive loss -------
            if self.supervised:
                pos_mask = self.gen_extra_positives_gt(target)
            else:
                index_matrix = repeat(indexes, "b -> c (d b)", c=n_augs * indexes.size(0), d=n_augs)
                pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
            neg_mask = (~pos_mask).fill_diagonal_(False)

            nce_loss = manual_simclr_loss_func(
                z,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
                temperature=self.temperature,
            )
        else:
            feats1, feats2 = out["feats"]

            z1 = self.projector(feats1)
            z2 = self.projector(feats2)

            # ------- contrastive loss -------
            if self.supervised:
                pos_mask = self.gen_extra_positives_gt(target)
                nce_loss = simclr_loss_func(
                    z1, z2, extra_pos_mask=pos_mask, temperature=self.temperature
                )
            else:
                nce_loss = simclr_loss_func(z1, z2, temperature=self.temperature)

        # compute number of extra positives
        n_positives = (
            (pos_mask != 0).sum().float()
            if self.supervised
            else torch.tensor(0.0, device=self.device)
        )

        metrics = {
            "train_nce_loss": nce_loss,
            "train_n_positives": n_positives,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

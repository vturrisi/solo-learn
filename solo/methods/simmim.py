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

import argparse
from typing import Any, Dict, List, Sequence


import torch
import torch.nn as nn
from solo.losses.simmim import simmim_loss_func
from solo.methods.base import BaseMethod


class SimMIM(BaseMethod):
    def __init__(
        self,
        mask_ratio: float,
        **kwargs,
    ):
        """Implements SimMIM (https://arxiv.org/abs/2111.09886).

        Args:
            mask_ratio (float): percentage of image to mask.
        """

        super().__init__(**kwargs)

        assert "vit" in self.backbone_name, "SimMIM only supports ViT as backbone atm."

        self.mask_ratio = mask_ratio
        self._vit_patch_size = self.backbone.patch_size

        stride = 16
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.features_dim, out_channels=stride**2 * 3, kernel_size=1),
            nn.PixelShuffle(stride),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimMIM, SimMIM).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov3")

        # parameters
        parser.add_argument("--mask_ratio", type=float, default=0.75)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.decoder.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def generate_mask(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # modified base forward
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        out = {}
        if self.training:
            mask = self.generate_mask(X)
            feats, patch_feats = self.backbone(X, mask)
            x_reconstructed = self.decoder(patch_feats)
            out.update({"mask": mask, "x_reconstructed": x_reconstructed})
        else:
            feats = self.backbone(X)

        logits = self.classifier(feats.detach())
        out.update({"logits": logits, "feats": feats})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MAE and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        patch_size = self._vit_patch_size
        imgs = batch[1]
        reconstruction_loss = 0
        for i in range(self.num_large_crops):
            reconstruction_loss += simmim_loss_func(
                x=imgs[i],
                x_rec=out["x_reconstructed"][i],
                mask=out["mask"][i],
                patch_size=patch_size,
            )
        reconstruction_loss /= self.num_large_crops

        metrics = {
            "train_reconstruction_loss": reconstruction_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return reconstruction_loss + class_loss

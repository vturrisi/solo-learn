import argparse
from typing import List

import torch.nn as nn
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseModel


class BarlowTwins(BaseModel):
    def __init__(
        self, proj_hidden_dim: int, output_dim: int, lamb: float, scale_loss: float, **kwargs
    ):
        super().__init__(**kwargs)

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwins, BarlowTwins).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=5e-3)
        parser.add_argument("--scale_loss", type=float, default=0.025)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """
        Adds projector parameters together with parent's learnable parameters.

        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch, batch_idx):
        """
        Training step for Barlow Twins reusing BaseModel training step.

        Args:
            batch: a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images
            batch_idx: index of the batch
        Returns:
            barlow loss + classification loss

        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- barlow twins loss -------
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        self.log("train_barlow_loss", barlow_loss, on_epoch=True, sync_dist=True)

        return barlow_loss + class_loss

# Copyright 2024 solo-learn development team.

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

import pickle
from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.nnclr import nnclr_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import gather, omegaconf_select
from solo.utils.momentum import initialize_momentum_params
from solo.utils.positional_encodings import PositionalEncodingPermute1D, Summer


class All4One(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.queue_size: int = cfg.method_kwargs.queue_size

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # second predictor
        self.predictor2 = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # internal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_output_dim,
            nhead=8,
            dim_feedforward=proj_output_dim * 2,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # positional encoder
        self.pos_enc = Summer(PositionalEncodingPermute1D(5))

        # queue
        self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # NN index queue
        self.register_buffer("queue_index", -torch.ones(self.queue_size, dtype=torch.long))

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(All4One, All4One).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        cfg.method_kwargs.queue_size = omegaconf_select(cfg, "method_kwargs.queue_size", 65536)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
            {"params": self.predictor2.parameters()},
            {"params": self.transformer_encoder.parameters(), "lr": 0.1},
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
    def dequeue_and_enqueue(self, z: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
            idx (torch.Tensor): batch of indexes
        """

        z = gather(z)
        y = gather(y)
        idx = gather(idx)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y  # type: ignore

        # NN indexes
        self.queue_index[ptr : ptr + batch_size] = idx

        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbors of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            torch.Tensor:
                indexes of the first NNs.
            torch.Tensor:
                extracted batch of NNs.
            torch.Tensor:
                NN indexes.
            torch.Tensor:
                NN labels.
        """

        idxx = (z @ self.queue.T).max(dim=1)[1]

        _, idx = (z @ self.queue.T).topk(5, dim=1)

        nn = self.queue[idx]
        nn_idx = self.queue_index[idx]
        nn_lb = self.queue_y[idx]

        return idxx, nn, nn_idx, nn_lb

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = F.normalize(self.momentum_projector(out["feats"]), dim=-1)
        out.update({"z": z})
        return out

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent, the projected features and the
                predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def off_diagonal(self, x):
        """Extracts off-diagonal elements.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            torch.Tensor:
                flattened off-diagonal elements.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def save_NN(self, img_indexes, nn1_idx, nn1_lb):
        """Auxiliar function to store the NNs.

        Args:
            img_indexes (torch.Tensor): batch of image indexes in tensor format.
            nn1_idx (torch.Tensor): batch of NN indexes in tensor format.
            nn1_lb (torch.Tensor): batch of NN labels in tensor format.

        """

        with open(f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__NNS.pickle", "wb") as f:
            pickle.dump(nn1_idx.cpu().numpy(), f)

        with open(f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__IDX.pickle", "wb") as f:
            pickle.dump(img_indexes.cpu().numpy(), f)

        with open(
            f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__Labels.pickle", "wb"
        ) as f:
            pickle.dump(nn1_lb.cpu().numpy(), f)

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for All4One reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of All4One and classification loss.
        """

        targets = batch[-1]
        img_indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_z1, momentum_z2 = out["momentum_z"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        p1_2 = self.predictor2(z1)
        p2_2 = self.predictor2(z2)

        # find nn
        idx1, nn1, *_ = self.find_nn(momentum_z1)
        _, nn2, _, _ = self.find_nn(momentum_z2)

        trans_emb1 = self.pos_enc(nn1)
        trans_emb2 = self.pos_enc(nn2)

        # Shift Operation
        strange1 = self.pos_enc(torch.cat((p1_2.unsqueeze(1), nn1), 1)[:, :5, :])
        strange2 = self.pos_enc(torch.cat((p2_2.unsqueeze(1), nn2), 1)[:, :5, :])

        # Feature dimension task
        p1_norm_feat = torch.nn.functional.normalize(momentum_z1, dim=0)
        p2_norm_feat = torch.nn.functional.normalize(momentum_z2, dim=0)
        z1_norm_feat = torch.nn.functional.normalize(z1, dim=0)
        z2_norm_feat = torch.nn.functional.normalize(z2, dim=0)

        corr_matrix_1_feat = p1_norm_feat.T @ z2_norm_feat
        corr_matrix_2_feat = p2_norm_feat.T @ z1_norm_feat

        on_diag_feat = (
            (
                torch.diagonal(corr_matrix_1_feat).add(-1).pow(2).mean()
                + torch.diagonal(corr_matrix_2_feat).add(-1).pow(2).mean()
            )
            * 0.5
        ).sqrt()
        off_diag_feat = (
            (
                self.off_diagonal(corr_matrix_1_feat).pow(2).mean()
                + self.off_diagonal(corr_matrix_2_feat).pow(2).mean()
            )
            * 0.5
        ).sqrt()

        rich_emb1 = self.transformer_encoder(trans_emb1)[:, 0, :]
        rich_emb2 = self.transformer_encoder(trans_emb2)[:, 0, :]

        strange_emb1 = self.transformer_encoder(strange1)[:, 0, :]
        strange_emb2 = self.transformer_encoder(strange2)[:, 0, :]

        # ------- contrastive loss -------
        att_nnclr_loss = (
            nnclr_loss_func(rich_emb1, strange_emb2) / 2
            + nnclr_loss_func(rich_emb2, strange_emb1) / 2
        )

        nnclr_loss = (
            nnclr_loss_func(nn1[:, 0, :], p2, temperature=self.temperature) / 2
            + nnclr_loss_func(nn2[:, 0, :], p1, temperature=self.temperature) / 2
        )

        feature_loss = (0.5 * on_diag_feat + 0.5 * off_diag_feat) * 10

        b = targets.size(0)

        final_losss = 0.5 * att_nnclr_loss + 0.5 * nnclr_loss + 0.5 * feature_loss

        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        self.dequeue_and_enqueue(momentum_z1, targets, img_indexes)

        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_comb_loss": final_losss,
            "train_nnclr_loss": nnclr_loss,
            "train_att_nnclr_loss": att_nnclr_loss,
            "train_feature_loss": feature_loss,
            "train_nn_acc": nn_acc,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return final_losss + class_loss

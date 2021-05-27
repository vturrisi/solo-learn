import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import BaseModel
except:
    from .base import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.simsiam import simsiam_loss_func
from utils.metrics import accuracy_at_k


class SimSiam(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        proj_hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim
        pred_hidden_dim = args.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

    @property
    def extra_learnable_params(self):
        return [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        out1 = self(X1)
        out2 = self(X2)

        z1 = out1["z"]
        z2 = out2["z"]
        p1 = out1["p"]
        p2 = out2["p"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # ------- contrastive loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        z_std = F.normalize(torch.cat((z1, z2), dim=0), dim=1).std(dim=0).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_class_loss": class_loss,
            "train_z_std": z_std,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

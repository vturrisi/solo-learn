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

from losses.vicreg import vicreg_loss_func
from utils.metrics import accuracy_at_k


class VICReg(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}]

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(features)
            return z, y

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # features, projector features, class
        z1, output1 = self(X1, classify_only=False)
        z2, output2 = self(X2, classify_only=False)

        # ------- contrastive loss -------
        args = self.args
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=args.sim_loss_weight,
            var_loss_weight=args.var_loss_weight,
            cov_loss_weight=args.cov_loss_weight,
        )

        # ------- classification loss -------
        output = torch.cat((output1, output2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = vicreg_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_vicreg_loss": vicreg_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

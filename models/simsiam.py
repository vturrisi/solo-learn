import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.neg_cosine_sim import negative_cosine_similarity
from utils.metrics import accuracy_at_k


class SimSiam(Model):
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

    def forward(self, X, classify_only=True):
        feat, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(feat)
            p = self.predictor(z)
            return z, p, y

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # features, projector features, class
        z1, p1, output = self(X1, classify_only=False)
        z2, p2, _ = self(X2, classify_only=False)

        # ------- contrastive loss -------
        neg_cos_sim = (
            negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2
        )

        # ------- classification loss -------
        # for datasets with unsupervised data
        index = target >= 0
        output = output[index]
        target = target[index]

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

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

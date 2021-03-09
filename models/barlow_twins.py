import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from utils.gather_layer import gather
from utils.barlow_twins_loss import barlow_twins_loss
from utils.metrics import accuracy_at_k


class BarlowTwins(Model):
    def __init__(self, args):
        super().__init__(args)

        projection_bn = args.projection_bn
        hidden_mlp = args.hidden_mlp
        output_dim = args.encoding_size
        assert output_dim > 0

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(num_out_filters * block.expansion, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, output_dim),
        )

    def forward(self, X, classify_only=True):
        if classify_only:
            return super()(X, classify_only=classify_only)
        else:
            features, y = super()(X, classify_only=classify_only)
            z = self.projection_head(features)
            return features, z, y

    def training_step(self, batch, batch_idx):
        indexes, (X_aug1, X_aug2), target = batch
        X = torch.cat((X_aug1, X_aug2), dim=0)

        # features, projection head features, class
        features, z, output = self(X, classify_only=False)

        z1, z2 = torch.chunk(z, 2)
        z1 = gather(z1)
        z2 = gather(z2)

        # ------- contrastive loss -------
        barlow_loss = barlow_twins_loss(z1, z2)

        # ------- classification loss -------
        output = torch.chunk(output, 2)[0]
        # for datasets with unsupervised data
        index = target >= 0
        output = output[index]
        target = target[index]

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = barlow_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_barlow_loss": barlow_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

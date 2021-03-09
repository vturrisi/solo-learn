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
    def training_step(self, batch, batch_idx):

        indexes, (X_aug1, X_aug2), target = batch
        X = torch.cat((X_aug1, X_aug2), dim=0)

        # features, projection head features, class
        features, z, output = self(X, classify_only=False)

        z1, z2 = torch.chunk(z, 2)
        z1 = gather(z1)
        z2 = gather(z2)

        # ------- contrastive loss -------
        nce_loss = barlow_twins_loss(z1, z2)

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
        loss = nce_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))
        # compute number of extra positives
        n_positives = (
            (pos_mask != 0).sum().float()
            if self.args.supervised
            else torch.tensor(0.0, device=self.device)
        )

        metrics = {
            "train_nce_loss": nce_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_n_positives": n_positives,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss
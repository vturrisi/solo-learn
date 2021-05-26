import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from base import BaseModel
except:
    from .base import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.simclr import simclr_loss_func, manual_simclr_loss_func
from utils.gather_layer import gather
from utils.metrics import accuracy_at_k


class SimCLR(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim

        self.temperature = args.temperature

        # projector
        if hidden_dim == 0:
            self.projector = nn.Linear(self.features_size, output_dim)
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.features_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(features)
            return z, y

    @torch.no_grad()
    def gen_extra_positives_gt(self, Y):
        if self.args.multicrop:
            n_augs = self.args.n_crops + self.args.n_small_crops
        else:
            n_augs = 2
        labels_matrix = repeat(Y, "b -> c (d b)", c=n_augs * Y.size(0), d=n_augs)
        labels_matrix = (labels_matrix == labels_matrix.t()).fill_diagonal_(False)
        return labels_matrix

    def training_step(self, batch, batch_idx):
        if self.args.multicrop:
            n_crops = self.args.n_crops
            n_small_crops = self.args.n_small_crops
            n_augs = n_crops + n_small_crops

            indexes, all_X, target = batch

            X = torch.cat(all_X[:n_crops], dim=0)
            X_small = torch.cat(all_X[n_crops:], dim=0)

            # projector features, class
            z, output = self(X, classify_only=False)
            z_small, _ = self(X_small, classify_only=False)

            z = [gather(part) for part in torch.chunk(z, n_crops)]
            z_small = [gather(part) for part in torch.chunk(z_small, n_small_crops)]
            z = torch.cat((*z, *z_small), dim=0)

            # ------- contrastive loss -------
            if self.args.supervised:
                gathered_target = gather(target)
                pos_mask = self.gen_extra_positives_gt(gathered_target)
            else:
                indexes = gather(indexes)
                index_matrix = repeat(indexes, "b -> c (d b)", c=n_augs * indexes.size(0), d=n_augs)
                pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
            neg_mask = (~pos_mask).fill_diagonal_(False)

            nce_loss = manual_simclr_loss_func(
                z, pos_mask=pos_mask, neg_mask=neg_mask, temperature=self.temperature,
            )
        else:
            indexes, (X_aug1, X_aug2), target = batch
            X = torch.cat((X_aug1, X_aug2), dim=0)

            # projector features, class
            z, output = self(X, classify_only=False)

            z1, z2 = torch.chunk(z, 2)
            z1 = gather(z1)
            z2 = gather(z2)

            # ------- contrastive loss -------
            if self.args.supervised:
                gathered_target = gather(target)
                pos_mask = self.gen_extra_positives_gt(gathered_target)
                nce_loss = simclr_loss_func(
                    z1, z2, extra_pos_mask=pos_mask, temperature=self.temperature
                )
            else:
                nce_loss = simclr_loss_func(z1, z2, temperature=self.temperature)

        # ------- classification loss -------
        target = target.repeat(2)
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

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

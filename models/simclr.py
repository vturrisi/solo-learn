import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import reduce, rearrange, repeat

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from utils.gather_layer import gather
from utils.info_nce import info_nce, manual_info_nce_sava
from utils.metrics import accuracy_at_k


class SimCLR(Model):
    @torch.no_grad()
    def gen_extra_positives_gt(self, Y):
        if self.args.multicrop:
            Y = Y.repeat(self.args.n_crops)
        else:
            Y = Y.repeat(2)
        b = Y.size(0)
        labels_matrix = Y.reshape(1, -1).repeat(b, 1)
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

            # features, projection head features, class
            features, z, output = self(X, classify_only=False)
            features_small, z_small, output_small = self(X_small, classify_only=False)

            z = [gather(part) for part in torch.chunk(z, n_crops)]
            z_small = [gather(part) for part in torch.chunk(z_small, n_small_crops)]
            z = torch.cat((*z, *z_small), dim=0)

            # ------- contrastive loss -------
            indexes = gather(indexes).repeat(n_augs)
            index_matrix = indexes.reshape(1, -1).repeat(indexes.size(0), 1)
            print(index_matrix)
            print(indexes.size())
            exit()
            pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
            negative_mask = (~pos_mask).fill_diagonal_(False)

            nce_loss = manual_info_nce_sava(
                z, pos_mask=pos_mask, negative_mask=negative_mask, temperature=self.temperature,
            )
        else:
            indexes, (X_aug1, X_aug2), target = batch
            X = torch.cat((X_aug1, X_aug2), dim=0)

            # features, projection head features, class
            features, z, output = self(X, classify_only=False)

            z1, z2 = torch.chunk(z, 2)
            z1 = gather(z1)
            z2 = gather(z2)

            # ------- contrastive loss -------
            if self.args.supervised:
                gathered_target = gather(target)
                pos_mask = self.gen_extra_positives_gt(gathered_target)
                nce_loss = info_nce(
                    z1, z2, extra_pos_mask=pos_mask, temperature=self.temperature
                )
            else:
                nce_loss = info_nce(z1, z2, temperature=self.temperature)

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

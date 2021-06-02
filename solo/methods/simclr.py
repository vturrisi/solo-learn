import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from solo.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k


class SimCLR(BaseModel):
    def __init__(self, output_dim, proj_hidden_dim, temperature, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature

        # projector
        if proj_hidden_dim == 0:
            self.projector = nn.Linear(self.features_size, output_dim)
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.features_size, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, output_dim),
            )

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}]

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
        return {**out, "z": z}

    @torch.no_grad()
    def gen_extra_positives_gt(self, Y):
        multicrop = self.extra_args["multicrop"]
        n_crops = self.extra_args["n_crops"]
        n_small_crops = self.extra_args["n_small_crops"]

        if multicrop:
            n_augs = n_crops + n_small_crops
        else:
            n_augs = 2
        labels_matrix = repeat(Y, "b -> c (d b)", c=n_augs * Y.size(0), d=n_augs)
        labels_matrix = (labels_matrix == labels_matrix.t()).fill_diagonal_(False)
        return labels_matrix

    def training_step(self, batch, batch_idx):
        if self.extra_args["multicrop"]:
            n_crops = self.extra_args["n_crops"]
            n_small_crops = self.extra_args["n_small_crops"]
            n_augs = n_crops + n_small_crops

            indexes, all_X, target = batch

            X = torch.cat(all_X[:n_crops], dim=0)
            X_small = torch.cat(all_X[n_crops:], dim=0)

            out = self(X)
            z = out["z"]
            logits = out["logits"]

            out_small = self(X_small)
            z_small = out_small["z"]

            z = torch.cat((z, z_small), dim=0)

            # ------- contrastive loss -------
            if self.extra_args["supervised"]:
                pos_mask = self.gen_extra_positives_gt(target)
            else:
                index_matrix = repeat(indexes, "b -> c (d b)", c=n_augs * indexes.size(0), d=n_augs)
                pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
            neg_mask = (~pos_mask).fill_diagonal_(False)

            nce_loss = manual_simclr_loss_func(
                z,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
                temperature=self.temperature,
            )
        else:
            indexes, (X1, X2), target = batch

            out1 = self(X1)
            out2 = self(X2)

            z1 = out1["z"]
            z2 = out2["z"]
            logits1 = out1["logits"]
            logits2 = out2["logits"]
            logits = torch.cat((logits1, logits2))

            # ------- contrastive loss -------
            if self.extra_args["supervised"]:
                pos_mask = self.gen_extra_positives_gt(target)
                nce_loss = simclr_loss_func(
                    z1, z2, extra_pos_mask=pos_mask, temperature=self.temperature
                )
            else:
                nce_loss = simclr_loss_func(z1, z2, temperature=self.temperature)

        # ------- classification loss -------
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = nce_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))
        # compute number of extra positives
        n_positives = (
            (pos_mask != 0).sum().float()
            if self.extra_args["supervised"]
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

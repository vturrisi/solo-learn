import torch
import torch.nn as nn
from einops import repeat
from solo.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from solo.methods.base import BaseModel


class SimCLR(BaseModel):
    def __init__(self, output_dim, proj_hidden_dim, temperature, supervised=False, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature
        self.supervised = supervised

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(SimCLR, SimCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # supervised-simclr
        parser.add_argument("--supervised", action="store_true")
        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    @torch.no_grad()
    def gen_extra_positives_gt(self, Y):
        if self.multicrop:
            n_augs = self.n_crops + self.n_small_crops
        else:
            n_augs = 2
        labels_matrix = repeat(Y, "b -> c (d b)", c=n_augs * Y.size(0), d=n_augs)
        labels_matrix = (labels_matrix == labels_matrix.t()).fill_diagonal_(False)
        return labels_matrix

    def training_step(self, batch, batch_idx):
        indexes, *_, target = batch

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        if self.multicrop:
            n_augs = self.n_crops + self.n_small_crops

            feats = out["feats"]

            z = torch.cat([self.projector(f) for f in feats])

            # ------- contrastive loss -------
            if self.supervised:
                pos_mask = self.gen_extra_positives_gt(target)
            else:
                index_matrix = repeat(indexes, "b -> c (d b)", c=n_augs * indexes.size(0), d=n_augs)
                pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
            neg_mask = (~pos_mask).fill_diagonal_(False)

            nce_loss = manual_simclr_loss_func(
                z, pos_mask=pos_mask, neg_mask=neg_mask, temperature=self.temperature,
            )
        else:
            feats1, feats2 = out["feats"]

            z1 = self.projector(feats1)
            z2 = self.projector(feats2)

            # ------- contrastive loss -------
            if self.supervised:
                pos_mask = self.gen_extra_positives_gt(target)
                nce_loss = simclr_loss_func(
                    z1, z2, extra_pos_mask=pos_mask, temperature=self.temperature
                )
            else:
                nce_loss = simclr_loss_func(z1, z2, temperature=self.temperature)

        # compute number of extra positives
        n_positives = (
            (pos_mask != 0).sum().float()
            if self.supervised
            else torch.tensor(0.0, device=self.device)
        )

        metrics = {
            "train_nce_loss": nce_loss,
            "train_n_positives": n_positives,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

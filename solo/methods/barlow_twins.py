import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k


class BarlowTwins(BaseModel):
    def __init__(self, proj_hidden_dim, output_dim, lamb, scale_loss, **kwargs):
        super().__init__(**kwargs)

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("barlow_twins")
        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=5e-3)
        parser.add_argument("--scale_loss", type=float, default=0.025)
        return parent_parser

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}]

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
        return {**out, "z": z}

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        out1 = self(X1)
        out2 = self(X2)

        z1 = out1["z"]
        z2 = out2["z"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # ------- contrastive loss -------
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = barlow_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        metrics = {
            "train_barlow_loss": barlow_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

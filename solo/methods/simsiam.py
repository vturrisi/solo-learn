import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseModel


class SimSiam(BaseModel):
    def __init__(
        self, output_dim, proj_hidden_dim, pred_hidden_dim, **kwargs,
    ):
        super().__init__(**kwargs)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # ------- contrastive loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss

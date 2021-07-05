import torch
import torch.nn as nn
from solo.losses.wmse import wmse_loss_func
from solo.utils.whitening import Whitening2d
from solo.methods.base import BaseModel


class WMSE(BaseModel):
    def __init__(
        self, output_dim, proj_hidden_dim, whitening_iters, whitening_size, whitening_eps, **kwargs
    ):
        super().__init__(**kwargs)

        self.whitening_iters = whitening_iters
        self.whitening_size = whitening_size

        assert self.whitening_size <= self.batch_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        self.whitening = Whitening2d(output_dim, eps=whitening_eps)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(WMSE, WMSE).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=1024)

        # wmse
        parser.add_argument("--whitening_iters", type=int, default=1)
        parser.add_argument("--whitening_size", type=int, default=256)
        parser.add_argument("--whitening_eps", type=float, default=0)

        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        v = self.projector(out["feats"])
        return {**out, "v": v}

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats = out["feats"]

        v = torch.cat([self.projector(f) for f in feats])

        # ------- wmse loss -------
        bs = self.batch_size
        num_losses, wmse_loss = 0, 0
        for _ in range(self.whitening_iters):
            z = torch.empty_like(v)
            perm = torch.randperm(bs).view(-1, self.whitening_size)
            for idx in perm:
                for i in range(self.n_crops):
                    z[idx + i * bs] = self.whitening(v[idx + i * bs]).type_as(z)
            for i in range(self.n_crops - 1):
                for j in range(i + 1, self.n_crops):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    wmse_loss += wmse_loss_func(x0, x1)
                    num_losses += 1
        wmse_loss /= num_losses

        self.log("train_neg_cos_sim", wmse_loss, on_epoch=True, sync_dist=True)

        return wmse_loss + class_loss

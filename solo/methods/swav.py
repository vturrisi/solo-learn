import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.swav import swav_loss_func
from solo.methods.base import BaseModel
from solo.utils.sinkhorn_knopp import SinkhornKnopp


class SwAV(BaseModel):
    def __init__(
        self,
        output_dim,
        proj_hidden_dim,
        num_prototypes,
        sk_iters,
        sk_epsilon,
        temperature,
        queue_size,
        epoch_queue_starts,
        freeze_prototypes_epochs,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.sk_iters = sk_iters
        self.sk_epsilon = sk_epsilon
        self.temperature = temperature
        self.queue_size = queue_size
        self.epoch_queue_starts = epoch_queue_starts
        self.freeze_prototypes_epochs = freeze_prototypes_epochs

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # prototypes
        self.prototypes = nn.utils.weight_norm(nn.Linear(output_dim, num_prototypes, bias=False))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(SwAV, SwAV).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("swav")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # queue settings
        parser.add_argument("--queue_size", default=3840, type=int)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--num_prototypes", type=int, default=3000)
        parser.add_argument("--sk_epsilon", type=float, default=0.05)
        parser.add_argument("--sk_iters", type=int, default=3)
        parser.add_argument("--freeze_prototypes_epochs", type=int, default=1)
        parser.add_argument("--epoch_queue_starts", type=int, default=15)
        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.prototypes.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def on_train_start(self):
        # sinkhorn-knopp needs the world size
        world_size = self.trainer.world_size if self.trainer else 1
        self.sk = SinkhornKnopp(self.sk_iters, self.sk_epsilon, world_size)
        # queue also needs the world size
        if self.queue_size > 0:
            self.register_buffer(
                "queue",
                torch.zeros(
                    2,
                    self.queue_size // world_size,
                    self.output_dim,
                    device=self.device,
                ),
            )

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        z = F.normalize(z)
        p = self.prototypes(z)
        return {**out, "z": z, "p": p}

    @torch.no_grad()
    def get_assignments(self, preds):
        bs = preds[0].size(0)
        assignments = []
        for i, p in enumerate(preds):
            # optionally use the queue
            if self.queue_size > 0 and self.current_epoch >= self.epoch_queue_starts:
                p_queue = self.prototypes(self.queue[i])
                p = torch.cat((p, p_queue))
            # compute assignments with sinkhorn-knopp
            assignments.append(self.sk(p)[:bs])
        return assignments

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = F.normalize(self.projector(feats1))
        z2 = F.normalize(self.projector(feats2))

        p1 = self.prototypes(z1)
        p2 = self.prototypes(z2)

        # ------- swav loss -------
        preds = [p1, p2]
        assignments = self.get_assignments(preds)
        swav_loss = swav_loss_func(preds, assignments, self.temperature)

        # ------- update queue -------
        if self.queue_size > 0:
            z = torch.stack((z1, z2))
            self.queue[:, z.size(1) :] = self.queue[:, : -z.size(1)].clone()
            self.queue[:, : z.size(1)] = z.detach()

        self.log("train_swav_loss", swav_loss, on_epoch=True, sync_dist=True)

        return swav_loss + class_loss

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for p in self.prototypes.parameters():
                p.grad = None

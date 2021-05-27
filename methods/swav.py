import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import BaseModel
except:
    from .base import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.swav import swav_loss_func
from utils.metrics import accuracy_at_k
from utils.sinkhorn_knopp import SinkhornKnopp


class SwAV(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        hidden_dim = args.hidden_dim
        num_prototypes = args.num_prototypes

        self.output_dim = args.encoding_dim
        self.sk_iters = args.sk_iters
        self.sk_epsilon = args.sk_epsilon
        self.temperature = args.temperature
        self.queue_size = args.queue_size
        self.epoch_queue_starts = args.epoch_queue_starts
        self.freeze_prototypes_epochs = args.freeze_prototypes_epochs

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # prototypes
        self.prototypes = nn.Linear(self.output_dim, num_prototypes, bias=False)
        self.normalize_prototypes()

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}, {"params": self.prototypes.parameters()}]

    def on_train_start(self):
        # sinkhorn-knopp needs the world size
        self.sk = SinkhornKnopp(self.sk_iters, self.sk_epsilon, self.trainer.world_size)
        # queue also needs the world size
        if self.queue_size > 0:
            self.register_buffer(
                "queue",
                torch.zeros(
                    2,
                    self.queue_size // self.trainer.world_size,
                    self.output_dim,
                    device=self.device,
                ),
            )

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, X, classify_only=True):
        feat, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(feat)
            z = F.normalize(z)
            p = self.prototypes(z)
            return p, z, y

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
        indexes, (X1, X2), target = batch

        p1, z1, output1 = self(X1, classify_only=False)
        p2, z2, output2 = self(X2, classify_only=False)

        # ------- swav loss -------
        preds = [p1, p2]
        assignments = self.get_assignments(preds)
        swav_loss = swav_loss_func(preds, assignments, self.temperature)

        # ------- classification loss -------
        output = torch.cat((output1, output2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = swav_loss + class_loss

        # ------- update queue -------
        if self.queue_size > 0:
            z = torch.stack((z1, z2))
            self.queue[:, z.size(1) :] = self.queue[:, : -z.size(1)].clone()
            self.queue[:, : z.size(1)] = z.detach()

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_ce_loss": swav_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for p in self.prototypes.parameters():
                p.grad = None

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.normalize_prototypes()

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.swav import swav_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k
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
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)
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

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
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
        indexes, (X1, X2), target = batch

        out1 = self(X1)
        out2 = self(X2)

        z1 = out1["z"]
        z2 = out2["z"]
        p1 = out1["p"]
        p2 = out2["p"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # ------- swav loss -------
        preds = [p1, p2]
        assignments = self.get_assignments(preds)
        swav_loss = swav_loss_func(preds, assignments, self.temperature)

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = swav_loss + class_loss

        # ------- update queue -------
        if self.queue_size > 0:
            z = torch.stack((z1, z2))
            self.queue[:, z.size(1) :] = self.queue[:, : -z.size(1)].clone()
            self.queue[:, : z.size(1)] = z.detach()

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

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

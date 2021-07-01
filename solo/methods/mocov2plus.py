import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.moco import moco_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.gather_layer import gather
from solo.utils.momentum import initialize_momentum_params


class MoCoV2Plus(BaseMomentumModel):
    def __init__(self, output_dim, proj_hidden_dim, temperature, queue_size, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(MoCoV2Plus, MoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self):
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        q = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "q": q}

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        feats1_momentum, feats2_momentum = out["feats_momentum"]

        q1 = self.projector(feats1)
        q2 = self.projector(feats2)
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)

        with torch.no_grad():
            k1 = self.momentum_projector(feats1_momentum)
            k2 = self.momentum_projector(feats2_momentum)
            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)

        # ------- contrastive loss -------
        # symmetric
        queue = self.queue.clone().detach()
        nce_loss = (
            moco_loss_func(q1, k2, queue[1], self.temperature)
            + moco_loss_func(q2, k1, queue[0], self.temperature)
        ) / 2

        # ------- update queue -------
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

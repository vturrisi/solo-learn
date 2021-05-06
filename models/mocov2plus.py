import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.gather_layer import gather
from utils.metrics import accuracy_at_k
from utils.momentum import initialize_momentum_params, MomentumUpdater
from losses.moco import moco_loss_func


class MoCoV2Plus(Model):
    def __init__(self, args):
        super().__init__(args)

        hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim
        assert output_dim > 0

        self.temperature = args.temperature
        self.queue_size = args.queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, self.features_size),
            nn.ReLU(),
            nn.Linear(self.features_size, output_dim),
        )

        # instantiate and initialize momentum encoder
        self.momentum_encoder = self.base_model(zero_init_residual=args.zero_init_residual)
        self.momentum_encoder.fc = nn.Identity()
        if args.cifar:
            self.momentum_encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # instantiate and initialize momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_size, self.features_size),
            nn.ReLU(),
            nn.Linear(self.features_size, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # momentum updater
        self.momentum_updater = MomentumUpdater(args.base_tau_momentum, args.final_tau_momentum)

        # create the queue
        self.register_buffer("queue", torch.randn(2, output_dim, args.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0,2,1)
        self.queue[:, :, ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(features)
            z = F.normalize(z)
            return z, y

    @torch.no_grad()
    def forward_momentum(self, X):
        features_momentum = self.momentum_encoder(X)
        z = self.projector(features_momentum)
        z = F.normalize(z)
        return z

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # forward online encoder
        q1, output1 = self(X1, classify_only=False)
        q2, output2 = self(X2, classify_only=False)

        # forward momentum encoder
        k1 = self.forward_momentum(X1)
        k2 = self.forward_momentum(X2)

        # ------- contrastive loss -------
        # symmetric
        queue = self.queue.clone().detach()
        nce_loss = (
            moco_loss_func(q1, k2, queue[0], self.temperature) + \
            moco_loss_func(q2, k1, queue[1], self.temperature)
        ) / 2

        # ------- classification loss -------
        output = torch.cat((output1, output2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = nce_loss + class_loss

        # ------- update queue -------
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_nce_loss": nce_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log tau momentum
        self.log("tau", self.momentum_updater.cur_tau)
        # update momentum encoder
        self.momentum_updater.update(
            online_nets=[self.encoder, self.projector],
            momentum_nets=[self.momentum_encoder, self.momentum_projector],
            cur_step=self.trainer.global_step,
            max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
        )

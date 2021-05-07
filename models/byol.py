import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.byol import byol_loss_func
from utils.metrics import accuracy_at_k
from utils.momentum import initialize_momentum_params, MomentumUpdater


class BYOL(Model):
    def __init__(self, args):
        super().__init__(args)

        proj_hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim
        pred_hidden_dim = args.pred_hidden_dim
        assert output_dim > 0

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
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
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # momentum updater
        self.momentum_updater = MomentumUpdater(args.base_tau_momentum, args.final_tau_momentum)

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(features)
            p = self.predictor(z)
            return z, p, y

    @torch.no_grad()
    def forward_momentum(self, X):
        features_momentum = self.momentum_encoder(X)
        z_momentum = self.momentum_projector(features_momentum)
        return z_momentum

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # forward online encoder
        z1, p1, output1 = self(X1, classify_only=False)
        z2, p2, output2 = self(X2, classify_only=False)

        # forward momentum encoder
        z1_momentum = self.forward_momentum(X1)
        z2_momentum = self.forward_momentum(X2)

        # ------- contrastive loss -------
        neg_cos_sim = byol_loss_func(p1, z2_momentum) / 2 + byol_loss_func(p2, z1_momentum) / 2

        # ------- classification loss -------
        # for datasets with unsupervised data
        output = torch.cat((output1, output2))
        target = target.repeat(2)
        index = target >= 0
        output = output[index]
        target = target[index]

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        z_std = F.normalize(torch.cat((z1, z2), dim=0), dim=1).std(dim=0).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_class_loss": class_loss,
            "train_z_std": z_std,
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

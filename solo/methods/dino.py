import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.dino import DINOLoss
from solo.methods.base import BaseMomentumModel
from solo.utils.momentum import initialize_momentum_params
from solo.utils.trunc_normal import trunc_normal_
import distutils


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=True,
        norm_last_layer=True,
        num_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINO(BaseMomentumModel):
    def __init__(
        self,
        output_dim,
        proj_hidden_dim,
        num_prototypes,
        norm_last_layer,
        clip_grad,
        freeze_last_layer,
        student_temperature,
        teacher_temperature,
        warmup_teacher_temperature,
        warmup_teacher_temperature_epochs,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer

        # dino head
        self.head = DINOHead(
            in_dim=self.features_size,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=output_dim,
            out_dim=num_prototypes,
            norm_last_layer=norm_last_layer,
        )

        # instantiate and initialize momentum dino head
        self.momentum_head = DINOHead(
            in_dim=self.features_size,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=output_dim,
            out_dim=num_prototypes,
            norm_last_layer=norm_last_layer,
        )
        initialize_momentum_params(self.head, self.momentum_head)

        # dino loss
        self.dino_loss_func = DINOLoss(
            out_dim=num_prototypes,
            student_temp=student_temperature,
            warmup_teacher_temp=warmup_teacher_temperature,
            teacher_temp=teacher_temperature,
            warmup_teacher_temp_epochs=warmup_teacher_temperature_epochs,
            num_epochs=self.max_epochs,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(DINO, DINO).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("dino")

        # optimization settings
        parser.add_argument("--clip_grad", type=float, default=0)
        parser.add_argument("--freeze_last_layer", type=int, default=1)

        # dino head
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--num_prototypes", type=int, default=4096)
        parser.add_argument("--norm_last_layer", type=distutils.util.strtobool, default=True)

        # temperature settings
        parser.add_argument("--student_temperature", type=float, default=0.1)
        parser.add_argument("--teacher_temperature", default=0.07, type=float)
        parser.add_argument("--warmup_teacher_temperature", default=0.04, type=float)
        parser.add_argument("--warmup_teacher_temperature_epochs", default=50, type=int)

        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [{"params": self.head.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self):
        extra_momentum_pairs = [(self.head, self.momentum_head)]
        return super().momentum_pairs + extra_momentum_pairs

    def clip_gradients(self, clip):
        for p in self.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self):
        self.dino_loss_func.epoch = self.current_epoch

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        p = self.head(out["feats"])
        return {**out, "p": p}

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        feats1_momentum, feats2_momentum = out["feats_momentum"]

        # forward online encoder
        p1 = self.head(feats1)
        p2 = self.head(feats2)
        p = torch.cat((p1, p2))

        # forward momentum encoder
        p1_momentum = self.momentum_head(feats1_momentum)
        p2_momentum = self.momentum_head(feats2_momentum)
        p_momentum = torch.cat((p1_momentum, p2_momentum))

        # ------- contrastive loss -------
        dino_loss = self.dino_loss_func(p, p_momentum)

        self.log("dino_loss", dino_loss, on_epoch=True, sync_dist=True)

        return dino_loss + class_loss

    def on_after_backward(self):
        # clip gradients
        if self.clip_grad:
            self.clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None

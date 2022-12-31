# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.dino import DINOLoss
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import omegaconf_select, trunc_normal_
from solo.utils.momentum import initialize_momentum_params


class DINOHead(nn.Module):
    mlp: Any
    last_layer: Any

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int,
        use_bn: bool = True,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        """DINO head that takes as input the features of the backbone, projects them in a lower
        dimensional space and multiplies with the prototypes.

        Args:
            in_dim (int): number of dimensions of the input (aka backbone features).
            num_prototypes (int): number of prototypes.
            use_bn (bool, optional): whether to use batch norm in projector. Defaults to True.
            norm_last_layer (bool, optional): whether to l2-norm the last layer. Defaults to True.
            num_layers (int, optional): number of layers in projector. Defaults to 3.
            hidden_dim (int, optional): number of dimension in hidden layers. Defaults to 2048.
            bottleneck_dim (int, optional): number of dimensions in bottleneck. Defaults to 256.
        """

        super().__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[Any] = [nn.Linear(in_dim, hidden_dim)]
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

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the backbone, the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINO(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Adds DINO head to the student and momentum DINO head to the teacher.

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                proj_output_dim (int): number of output neurons in the projector.
                num_prototypes (int): number of prototypes.
                use_bn_in_head (bool): whether or not to use bn in the head.
                norm_last_layer (bool): whether or not to normalize the last layer (prototypes).
                clip_grad (float): threshold for gradient clipping.
                freeze_last_layer (bool): whether or not to freeze the last layer (prototypes).
                student_temperature (float): temperature for the student.
                teacher_temperature (float): temperature for the teacher.
                warmup_teacher_temperature (float): base temperature for the teacher.
                warmup_teacher_temperature_epochs (int): number of epochs of cosine annealing
                    scheduling for teacher temperature.
        """

        super().__init__(cfg)

        self.clip_grad: bool = cfg.method_kwargs.clip_grad
        self.freeze_last_layer: bool = cfg.method_kwargs.freeze_last_layer

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        use_bn_in_head: bool = cfg.method_kwargs.use_bn_in_head
        num_prototypes: int = cfg.method_kwargs.num_prototypes
        norm_last_layer: bool = cfg.method_kwargs.norm_last_layer
        student_temperature: float = cfg.method_kwargs.student_temperature
        warmup_teacher_temperature: float = cfg.method_kwargs.warmup_teacher_temperature
        teacher_temperature: float = cfg.method_kwargs.teacher_temperature
        warmup_teacher_temperature_epochs: int = cfg.method_kwargs.warmup_teacher_temperature_epochs

        # dino head
        self.head = DINOHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            use_bn=use_bn_in_head,
            bottleneck_dim=proj_output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )

        # instantiate and initialize momentum dino head
        self.momentum_head = DINOHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            use_bn=use_bn_in_head,
            bottleneck_dim=proj_output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )
        initialize_momentum_params(self.head, self.momentum_head)

        # dino loss
        self.dino_loss_func = DINOLoss(
            num_prototypes=num_prototypes,
            student_temp=student_temperature,
            warmup_teacher_temp=warmup_teacher_temperature,
            teacher_temp=teacher_temperature,
            warmup_teacher_temp_epochs=warmup_teacher_temperature_epochs,
            num_epochs=self.max_epochs,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(DINO, DINO).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.num_prototypes")

        # optimization settings
        cfg.method_kwargs.clip_grad = omegaconf_select(cfg, "method_kwargs.clip_grad", 0)
        cfg.method_kwargs.freeze_last_layer = omegaconf_select(
            cfg, "method_kwargs.freeze_last_layer", 1
        )

        # head settings
        cfg.method_kwargs.norm_last_layer = omegaconf_select(
            cfg, "method_kwargs.norm_last_layer", True
        )
        cfg.method_kwargs.use_bn_in_head = omegaconf_select(
            cfg, "method_kwargs.use_bn_in_head", False
        )

        # temperature settings
        cfg.method_kwargs.student_temperature = omegaconf_select(
            cfg, "method_kwargs.student_temperature", 0.1
        )
        cfg.method_kwargs.teacher_temperature = omegaconf_select(
            cfg, "method_kwargs.teacher_temperature", 0.07
        )
        cfg.method_kwargs.warmup_teacher_temperature = omegaconf_select(
            cfg, "method_kwargs.warmup_teacher_temperature", 0.04
        )
        cfg.method_kwargs.warmup_teacher_temperature_epochs = omegaconf_select(
            cfg, "method_kwargs.warmup_teacher_temperature_epochs", 0
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds DINO head parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "head", "params": self.head.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (head, momentum_head) to the parent's momentum pairs.

        Returns:
            List[dict]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.head, self.momentum_head)]
        return super().momentum_pairs + extra_momentum_pairs

    def dino_clip_gradients(self, clip: float):
        """Clips gradients after backward pass.

        Args:
            clip (float): threshold for gradient clipping.
        """

        for p in self.backbone.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self):
        """Updates the current epoch in DINO's loss object."""
        self.dino_loss_func.epoch = self.current_epoch

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the student (backbone and head).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X)
        z = self.head(out["feats"])
        out.update({"z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the key.
        """

        out = super().momentum_forward(X)
        z = self.momentum_head(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DINO reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where [X]
                is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DINO loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        p = torch.cat(out["z"])
        momentum_p = torch.cat(out["momentum_z"])

        # ------- contrastive loss -------
        dino_loss = self.dino_loss_func(p, momentum_p)

        self.log("dino_loss", dino_loss, on_epoch=True, sync_dist=True)

        return dino_loss + class_loss

    def on_after_backward(self):
        """Performs gradient clipping and zeros the gradients on the last layer (prototypes)."""

        # clip gradients
        if self.clip_grad:
            self.dino_clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None

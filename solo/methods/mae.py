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

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.mae import mae_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import generate_2d_sincos_pos_embed, omegaconf_select
from timm.models.vision_transformer import Block


class MAEDecoder(nn.Module):
    def __init__(
        self, in_dim, embed_dim, depth, num_heads, num_patches, patch_size, mlp_ratio=4.0
    ) -> None:
        super().__init__()

        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)

        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class MAE(BaseMethod):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Implements MAE (https://arxiv.org/abs/2111.06377).

        Extra cfg settings:
            method_kwargs:
                mask_ratio (float): percentage of image to mask.
                decoder_embed_dim (int): number of dimensions for the embedding in the decoder
                decoder_depth (int) depth of the decoder
                decoder_num_heads (int) number of heads for the decoder
                norm_pix_loss (bool): whether to normalize the pixels of each patch with their
                    respective mean and std for the loss. Defaults to False.
        """

        super().__init__(cfg)

        assert "vit" in self.backbone_name, "MAE only supports ViT as backbone."

        self.mask_ratio: float = cfg.method_kwargs.mask_ratio
        self.norm_pix_loss: bool = cfg.method_kwargs.norm_pix_loss

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        # if patch size is not available, defaults to 16 or 14 depending on backbone
        default_patch_size = 14 if self.backbone_name == "vit_huge" else 16
        self._vit_patch_size: int = self.backbone_args.get("patch_size", default_patch_size)
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        decoder_embed_dim: int = cfg.method_kwargs.decoder_embed_dim
        decoder_depth: int = cfg.method_kwargs.decoder_depth
        decoder_num_heads: int = cfg.method_kwargs.decoder_num_heads

        # decoder
        self.decoder = MAEDecoder(
            in_dim=self.features_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            num_patches=self._vit_num_patches,
            patch_size=self._vit_patch_size,
            mlp_ratio=4.0,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(MAE, MAE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.decoder_embed_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.decoder_depth")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.decoder_num_heads")

        cfg.method_kwargs.mask_ratio = omegaconf_select(cfg, "method_kwargs.mask_ratio", 0.75)
        cfg.method_kwargs.norm_pix_loss = omegaconf_select(
            cfg,
            "method_kwargs.norm_pix_loss",
            False,
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "decoder", "params": self.decoder.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # modified base forward
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        out = {}
        if self.training:
            feats, patch_feats, mask, ids_restore = self.backbone(X, self.mask_ratio)
            pred = self.decoder(patch_feats, ids_restore)
            out.update({"mask": mask, "pred": pred})
        else:
            feats = self.backbone(X)

        logits = self.classifier(feats.detach())
        out.update({"logits": logits, "feats": feats})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MAE and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        patch_size = self._vit_patch_size
        imgs = batch[1]
        reconstruction_loss = 0
        for i in range(self.num_large_crops):
            reconstruction_loss += mae_loss_func(
                imgs[i],
                out["pred"][i],
                out["mask"][i],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
            )
        reconstruction_loss /= self.num_large_crops

        metrics = {
            "train_reconstruction_loss": reconstruction_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return reconstruction_loss + class_loss

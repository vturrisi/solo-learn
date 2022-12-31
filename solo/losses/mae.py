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

import torch


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchifies an image according to some patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        patch_size (int): size of each patch.

    Returns:
        torch.Tensor: [N, Tokens, pixels * pixels * 3] Tensor containing the patchified images.
    """

    assert imgs.size(2) == imgs.size(3) and imgs.size(2) % patch_size == 0

    h = w = imgs.size(2) // patch_size
    x = imgs.reshape(shape=(imgs.size(0), 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.size(0), h * w, patch_size**2 * 3))
    return x


def mae_loss_func(
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """Computes MAE's loss given batch of images, the decoder predictions, the input mask and respective patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        pred (torch.Tensor): [N, Tokens, pixels * pixels * 3] Tensor containing the predicted patches.
        mask (torch.Tensor): [N, Tokens] Tensor representing a binary mask, where value 1 means masked.
        patch_size (int): size of each patch.
        norm_pix_loss (bool): whether to normalize the pixels of each patch with their respective mean and std.

    Returns:
        torch.Tensor: MAE's loss.
    """

    target = patchify(imgs, patch_size)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

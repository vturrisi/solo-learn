# Copyright 2022 solo-learn development team.

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
import torch.nn.functional as F


def simmim_loss_func(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    input_channels: int = 3,
):
    """Computes SimMIM's loss given batch of reconstructed images and images.

    Args:
        x (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        x_rec (torch.Tensor): [N, 3, H, W] Tensor containing the reconstructed images.
        mask (torch.Tensor): [N, Tokens] Tensor representing a binary mask, where value 1 means masked.
        patch_size (int): size of the individual patches.
        input_channel (int): number of input channels of the images. Defaults to 3.

    Returns:
        torch.Tensor: SimMIM's loss.
    """

    mask = (
        mask.repeat_interleave(patch_size, 1)
        .repeat_interleave(patch_size, 2)
        .unsqueeze(1)
        .contiguous()
    )
    loss_recon = F.l1_loss(x, x_rec, reduction="none")
    loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / input_channels
    return loss

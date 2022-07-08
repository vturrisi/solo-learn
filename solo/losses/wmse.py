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


def wmse_loss_func(z1: torch.Tensor, z2: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes W-MSE's loss given two batches of whitened features z1 and z2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing whitened features from view 1.
        z2 (torch.Tensor): NxD Tensor containing whitened features from view 2.
        simplified (bool): faster computation, but with same result.

    Returns:
        torch.Tensor: W-MSE loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(z1, z2.detach(), dim=-1).mean()

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

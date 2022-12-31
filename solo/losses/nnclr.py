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
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank


def nnclr_loss_func(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)
    # to be consistent with simclr, we now gather p
    # this might result in suboptimal results given previous parameters.
    p = gather(p)

    logits = nn @ p.T / temperature

    rank = get_rank()
    n = nn.size(0)
    labels = torch.arange(n * rank, n * (rank + 1), device=p.device)
    loss = F.cross_entropy(logits, labels)
    return loss

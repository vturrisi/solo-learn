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


def mean_shift_loss_func(
    p: torch.Tensor, z: torch.Tensor, queue: torch.Tensor, num_neighbors: int
) -> torch.Tensor:
    """Computes meanshift's loss given a batch of predicted features p
    and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        queue: QxD Tensor containing momentum features obtained from previous iterations
        num_neighbors: Number of nearest neighbors

    Returns:
        torch.Tensor: Mean shift loss.
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    queue = F.normalize(queue, dim=1)
    _, indices = torch.topk(z @ queue.T, num_neighbors, dim=1)
    nn_targets = queue[indices]
    z = z.unsqueeze(dim=1)
    p = p.unsqueeze(dim=1)
    return 2 - 2 * (torch.einsum("nik,njk->nj", p, nn_targets.detach())).mean()

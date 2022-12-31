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
import torch.distributed as dist
import torch.nn.functional as F
from solo.utils.misc import concat_all_gather_no_grad


def mocov3_loss_func(query: torch.Tensor, key: torch.Tensor, temperature=0.2) -> torch.Tensor:
    """Computes MoCo V3's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the keys from view 2.
        temperature (float, optional): temperature of the softmax in the contrastive
            loss. Defaults to 0.2.

    Returns:
        torch.Tensor: MoCo loss.
    """

    n = query.size(0)
    device = query.device
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)

    # gather all targets without gradients
    key = concat_all_gather_no_grad(key)

    logits = torch.einsum("nc,mc->nm", [query, key]) / temperature
    labels = torch.arange(n, dtype=torch.long, device=device) + n * rank

    return F.cross_entropy(logits, labels) * (2 * temperature)

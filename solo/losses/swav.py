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

from typing import List

import numpy as np
import torch


def swav_loss_func(
    preds: List[torch.Tensor], assignments: List[torch.Tensor], temperature: float = 0.1
) -> torch.Tensor:
    """Computes SwAV's loss given list of batch predictions from multiple views
    and a list of cluster assignments from the same multiple views.

    Args:
        preds (torch.Tensor): list of NxC Tensors containing nearest neighbors' features from
            view 1.
        assignments (torch.Tensor): list of NxC Tensor containing predicted features from view 2.
        temperature (torch.Tensor): softmax temperature for the loss. Defaults to 0.1.

    Returns:
        torch.Tensor: SwAV loss.
    """

    losses = []
    for v1, a in enumerate(assignments):
        for v2 in np.delete(np.arange(len(preds)), v1):
            p = preds[v2] / temperature
            loss = -torch.mean(torch.sum(a * torch.log_softmax(p, dim=1), dim=1))
            losses.append(loss)
    return sum(losses) / len(losses)

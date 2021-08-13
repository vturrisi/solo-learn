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
    for v1 in range(len(preds)):
        for v2 in np.delete(np.arange(len(preds)), v1):
            a = assignments[v1]
            p = preds[v2] / temperature
            loss = -torch.mean(torch.sum(a * torch.log_softmax(p, dim=1), dim=1))
            losses.append(loss)
    return sum(losses) / len(losses)

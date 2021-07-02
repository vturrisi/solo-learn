import numpy as np
import torch


def swav_loss_func(
    preds: list[torch.Tensor], assignments: list[torch.Tensor], temperature: float = 0.1
):
    """
    Applies SWaV's loss given list of batch predictions from multiple views
    and a list of cluster assignments from the same multiple views.

    Args:
        preds: list of NxC Tensors containing nearest neighbors' features from view 1
        assignments: list of NxC Tensor containing predicted features from view 2
        temperature: temperature factor for the loss

    """

    losses = []
    for v1 in range(len(preds)):
        for v2 in np.delete(np.arange(len(preds)), v1):
            a = assignments[v1]
            p = preds[v2] / temperature
            loss = -torch.mean(torch.sum(a * torch.log_softmax(p, dim=1), dim=1))
            losses.append(loss)
    return sum(losses) / len(losses)

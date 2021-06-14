import numpy as np
import torch


def swav_loss_func(preds, assignments, temperature=0.1):
    losses = []
    for v1 in range(len(preds)):
        for v2 in np.delete(np.arange(len(preds)), v1):
            a = assignments[v1]
            p = preds[v2] / temperature
            loss = -torch.mean(torch.sum(a * torch.log_softmax(p, dim=1), dim=1))
            losses.append(loss)
    return sum(losses) / len(losses)

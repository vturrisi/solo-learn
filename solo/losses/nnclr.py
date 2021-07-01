import torch
import torch.nn.functional as F


def nnclr_loss_func(nn, p, temperature=0.1):
    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)

    logits = nn @ p.T / temperature

    n = p.size(0)
    labels = torch.arange(n, device=p.device)

    loss = F.cross_entropy(logits, labels)
    return loss

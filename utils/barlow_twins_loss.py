import torch
import torch.nn.functional as F
from einops import rearrange, reduce


def barlow_twins_loss(z1, z2, l=5e-3):
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    N, D = z1.size()

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()].mul_(l)
    loss = cdif.sum()
    return loss

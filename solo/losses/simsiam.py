import torch
import torch.nn.functional as F


def simsiam_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True):
    """
    Applies SimSiam's loss given batch of predicted features p from view 1 and
    a batch of projected features z from view 2.

    Args:
        p: Tensor containing predicted features from view 1
        z: Tensor containing projected features from view 2
        simplified: faster computation, but with same result

    """

    if simplified:
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return -(p * z.detach()).sum(dim=1).mean()

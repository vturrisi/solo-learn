import torch
import torch.nn.functional as F


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True):
    """
    Applies BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p: NxD Tensor containing predicted features from view 1
        z: NxD Tensor containing projected momentum features from view 2
        simplified: faster computation, but with same result

    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()

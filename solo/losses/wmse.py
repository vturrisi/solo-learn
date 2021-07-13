import torch
import torch.nn.functional as F


def wmse_loss_func(z1: torch.Tensor, z2: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes W-MSE's loss given two batches of whitened features z1 and z2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing whitened features from view 1.
        z2 (torch.Tensor): NxD Tensor containing whitened features from view 2.
        simplified (bool): faster computation, but with same result.

    Returns:
        torch.Tensor: W-MSE loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(z1, z2.detach(), dim=-1).mean()
    else:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

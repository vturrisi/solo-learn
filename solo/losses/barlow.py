import torch

import torch.distributed as dist


def barlow_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025
) -> torch.Tensor:
    """Computes Barlow Twins' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

    Returns:
        torch.Tensor: Barlow Twins' loss.
    """

    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(corr)
        world_size = dist.get_world_size()
        corr /= world_size

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss

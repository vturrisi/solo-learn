import torch


def barlow_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025
):
    """
    Applies Barlow Twins' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1: NxD Tensor containing projected features from view 1
        z2: NxD Tensor containing projected features from view 2
        lamb: off-diagonal scaling factor for the cross-covariance matrix
        scale_loss: final scaling factor of the loss
    """

    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss

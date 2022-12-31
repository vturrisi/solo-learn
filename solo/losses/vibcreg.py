# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from solo.losses.vicreg import invariance_loss, variance_loss
from solo.utils.misc import gather


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes normalized covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    norm_z1 = z1 - z1.mean(dim=0)
    norm_z2 = z2 - z2.mean(dim=0)
    norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
    norm_z2 = F.normalize(norm_z2, p=2, dim=0)
    fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
    fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
    fxf_cov_z1.fill_diagonal_(0.0)
    fxf_cov_z2.fill_diagonal_(0.0)
    cov_loss = (fxf_cov_z1**2).mean() + (fxf_cov_z2**2).mean()
    return cov_loss


def vibcreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 200.0,
) -> torch.Tensor:
    """Computes VIbCReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VIbCReg loss.
    """

    sim_loss = invariance_loss(z1, z2)
    # vicreg's official coded gathers the tensors here, so it's likely to benefit vibcreg
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    z1, z2 = gather(z1), gather(z2)

    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    return loss

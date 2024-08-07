# Copyright 2024 solo-learn development team.

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

from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F

def calculate_frobenius_regularization_term(z: torch.Tensor) -> torch.Tensor:
    V, N, D = z.shape

    if N > D:
        cov = torch.matmul(z.transpose(1, 2), z)  # V x D x D
    else:
        cov = torch.matmul(z, z.transpose(1, 2))  # V x N x N
    
    # divide each view covariance by its trace
    trace = torch.diagonal(cov, dim1=1, dim2=2)    # V x D
    trace = torch.sum(trace, dim=1)                # V x 1
    cov = cov / trace.unsqueeze(-1).unsqueeze(-1) 

    # REGULARIZATION TERM - sum the log-frobenius norm of each view covariance matrix
    fro_norm_per_view = torch.linalg.norm(cov, dim=(1,2), ord='fro')  # V x 1
    regularization_term = -torch.sum( 2*torch.log(fro_norm_per_view) ) # we bring frobenius square outside log

    return regularization_term

def calculate_invariance_term(z: torch.Tensor) -> torch.Tensor:
    V, N, D = z.shape

    # INVARIANCE - align each view to the average view
    average_z = torch.mean(z, dim=0)       # N x D, samples are averaged across views
    average_z = average_z.repeat(V, 1, 1)  # V x N x D
    invariance_loss_term = F.mse_loss(z, average_z)

    return invariance_loss_term

def frossl_loss_func(
    z: torch.Tensor, invariance_weight=1, logger=None
) -> torch.Tensor:
    """
    Implements FroSSL (https://arxiv.org/pdf/2310.02903)
    Heavily adapted from https://github.com/OFSkean/FroSSL. The main difference is that this
    implementation stacks the views and operates on all of them at once, rather than one at a time.
    This saves ~2 seconds (about 5% improvement) per batch with N=2,D=1024 on a A5000 GPU. For a simpler,
    ableit slower, implementation of loss that operates on one view at a time, please see 
    the original implementation.

    Args:
        z (torch.Tensor): V x N x D Tensor containing projected features from the views.
            Every N-th sample is a different view of the same image.
        invariance_weight (float): weight for the invariance loss term. default is 1.

    Return:
        torch.Tensor: FroSSL loss.
    """
    V, N, D = z.shape

    z = F.normalize(z, dim=1)  # V x N x D

    regularization_term = calculate_frobenius_regularization_term(z)
    regularization_term = -1 * regularization_term # make sure its maximized

    invariance_tradeoff = V * D * invariance_weight
    invariance_term = calculate_invariance_term(z)
    invariance_term = invariance_tradeoff * invariance_term

    if logger is not None:
        logger("frossl_regularization_loss", -regularization_term, sync_dist=True)
        logger("frossl_invariance_loss", invariance_term, sync_dist=True)

    total_loss = regularization_term + invariance_term
    return total_loss
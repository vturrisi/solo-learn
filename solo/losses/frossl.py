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

from typing import List, Tuple

import torch
import torch.nn.functional as F


def frossl_loss_func(
    z: List[torch.Tensor], invariance_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the FroSSL loss given a list of projected views
    (https://arxiv.org/abs/2310.02903).

    The loss is the sum, over every view, of a Frobenius-norm regularization term
    (which maximizes the entropy of the view's covariance/gram matrix) and an
    invariance term (which pulls the views towards their mean). It supports any
    number of views V >= 2.

    Args:
        z (List[torch.Tensor]): list of V NxD Tensors containing projected features,
            one per view.
        invariance_weight (float): weight of the invariance term. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total loss, invariance term and regularization term (summed over views).
            The individual terms are returned so that the method can log them, keeping
            this function side-effect free like the other loss functions in solo-learn.
    """

    N = z[0].size(0)
    D = z[0].size(1)
    V = len(z)

    # normalize each view along the batch dimension
    normalized_z = [F.normalize(zv, p=2, dim=0) for zv in z]
    average_embedding = torch.mean(torch.stack(normalized_z), dim=0)

    total_invariance = z[0].new_zeros(())
    total_regularization = z[0].new_zeros(())
    for view_embeddings in normalized_z:
        # regularization term (eq. 6 in the paper): maximize the entropy of the
        # (auto)covariance matrix, estimated through its squared Frobenius norm.
        if N > D:
            cov = view_embeddings.T @ view_embeddings
        else:
            cov = view_embeddings @ view_embeddings.T
        # .item() avoids a float16 casting issue when dividing by the trace
        cov = cov / torch.trace(cov).item()
        fro_norm = torch.linalg.norm(cov, ord="fro")
        # the 2 brings the frobenius square outside of the log
        regularization = 2 * torch.log(fro_norm)
        total_regularization = total_regularization + regularization

        # invariance term (eq. 3 in the paper): pull each view towards the mean view
        invariance = V * D * F.mse_loss(view_embeddings, average_embedding)
        total_invariance = total_invariance + invariance

    total_invariance = invariance_weight * total_invariance
    loss = total_regularization + total_invariance
    return loss, total_invariance, total_regularization

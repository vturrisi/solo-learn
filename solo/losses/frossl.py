from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F

def frossl_loss_func(
    z: torch.Tensor, invariance_weight=1
) -> torch.Tensor:
    """Computes FroSSL's loss given batch of projected features z
    from num_crops different views.

    Args:
        z (torch.Tensor): views x N x D Tensor containing projected features from the views.
            Every Nth sample is a different view of the same image.
        invariance_weight (float): weight for the invariance loss term. default is 1.

    Return:
        torch.Tensor: FroSSL loss.
    """
    V, N, D = z.shape

    z = F.normalize(z, dim=-1)  # V x N x D

    if N > D:
        cov = view_embeddings.T @ view_embeddings # V x D x D
    else:
        cov = view_embeddings @ view_embeddings.T # V x N x N
    cov = cov / torch.trace(cov)

    # sum the log-frobenius norm of each view covariance matrix
    fro_norm_per_view = torch.linalg.norm(cov, ord='fro') # V x 1
    regularization_term = torch.sum( -2*torch.log(fro_norm) ) # bring frobenius square outside log

    # align each view to the average view
    average_z = torch.mean(z, dim=0) # N x D, samples are averaged across views
    invariance_loss_term = F.mse_loss(z, average_z)

    total_loss = regularization_term + invariance_weight*invariance_loss_term
    return total_loss
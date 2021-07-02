import torch
import torch.nn.functional as F


def moco_loss_func(query, key, queue, temperature=0.1):
    """
    Applies MoCo's loss given a batch of queries from view 1,
    a batch of keys from view 2 and a queue of past elements.

    Args:
        preds: list of NxC Tensors containing nearest neighbors' features from view 1
        assignments: list of NxC Tensor containing predicted features from view 2
        temperature: temperature factor for the loss

    """

    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)

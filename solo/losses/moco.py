import torch
import torch.nn.functional as F


def moco_loss_func(
    query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor, temperature=0.1
) -> torch.Tensor:
    """Computes MoCo's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): [description]. temperature of the softmax in the contrastive
            loss. Defaults to 0.1.

    Returns:
        torch.Tensor: MoCo loss.
    """

    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)

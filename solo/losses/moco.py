import torch
import torch.nn.functional as F


def moco_loss_func(query, key, queue, temperature=0.1):
    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)

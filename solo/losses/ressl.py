import torch
import torch.nn.functional as F


def ressl_loss_func(
    q: torch.Tensor,
    k: torch.Tensor,
    queue: torch.Tensor,
    temperature_q: float = 0.1,
    temperature_k: float = 0.04,
) -> torch.Tensor:
    """Computes ReSSL's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature_q (float, optional): [description]. temperature of the softmax for the query.
            Defaults to 0.1.
        temperature_k (float, optional): [description]. temperature of the softmax for the key.
            Defaults to 0.04.

    Returns:
        torch.Tensor: ReSSL loss.
    """

    logits_q = torch.einsum("nc,kc->nk", [q, queue])
    logits_k = torch.einsum("nc,kc->nk", [k, queue])

    loss = -torch.sum(
        F.softmax(logits_k.detach() / temperature_k, dim=1)
        * F.log_softmax(logits_q / temperature_q, dim=1),
        dim=1,
    ).mean()

    return loss

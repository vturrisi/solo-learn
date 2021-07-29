import torch
import torch.nn.functional as F


def ressl_loss_func(
    q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float = 0.04
) -> torch.Tensor:
    """Computes ReSSL's loss given a batch of queries from view 1 and
    a batch of logits_keys from view 2.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): [description]. temperature of the softmax for the logits_key.
            Defaults to 0.04.

    Returns:
        torch.Tensor: ReSSL loss.
    """

    logits_q = torch.einsum("nc,kc->nk", [q, queue])
    logits_k = torch.einsum("nc,kc->nk", [k, queue])

    loss = -torch.sum(
        F.softmax(logits_k.detach() / temperature, dim=1) * F.log_softmax(logits_q / 0.1, dim=1),
        dim=1,
    ).mean()

    return loss

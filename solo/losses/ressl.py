import torch
import torch.nn.functional as F


def ressl_loss_func(
    logits_query: torch.Tensor, logits_key: torch.Tensor, temperature: float = 0.4
) -> torch.Tensor:
    """Computes ReSSL's loss given a batch of queries from view 1 and a batch of logits_keys from view 2.

    Args:
        logits_query (torch.Tensor): NxD Tensor containing the queries from view 1.
        logits_key (torch.Tensor): NxD Tensor containing the queries from view 2.
        temperature (float, optional): [description]. temperature of the softmax for the logits_key.
            Defaults to 0.4.

    Returns:
        torch.Tensor: ReSSL loss.
    """

    loss = -torch.sum(
        F.softmax(logits_key.detach() / temperature, dim=1)
        * F.log_softmax(logits_query / 0.1, dim=1),
        dim=1,
    ).mean()
    return loss

import torch
import torch.nn.functional as F


def nnclr_loss_func(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)

    logits = nn @ p.T / temperature

    n = p.size(0)
    labels = torch.arange(n, device=p.device)

    loss = F.cross_entropy(logits, labels)
    return loss

import torch
import torch.nn.functional as F


def deepclusterv2_loss_func(
    outputs: torch.Tensor, assignments: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes DeepClusterV2's loss given a tensor containing logits from multiple views
    and a tensor containing cluster assignments from the same multiple views.

    Args:
        outputs (torch.Tensor): tensor of size PxVxNxC where P is the number of prototype
            layers and V is the number of views.
        assignments (torch.Tensor): tensor of size PxVxNxC containing the assignments
            generated using k-means.
        temperature (float, optional): softmax temperature for the loss. Defaults to 0.1.

    Returns:
        torch.Tensor: DeepClusterV2 loss.
    """
    loss = 0
    for h in range(outputs.size(0)):
        scores = outputs[h].view(-1, outputs.size(-1)) / temperature
        targets = assignments[h].repeat(outputs.size(1)).to(outputs.device, non_blocking=True)
        loss += F.cross_entropy(scores, targets, ignore_index=-1)
    return loss / outputs.size(0)

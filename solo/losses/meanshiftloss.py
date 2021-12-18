import torch
import torch.nn.functional as F

def mean_shift_loss_func(p: torch.Tensor,
        z: torch.Tensor,
        queue: torch.Tensor,
        num_neighbors: int) -> torch.Tensor:
    """Computes mean shift loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        queue: QxD Tensor containing momentum features obtained from previous iterations
        num_neighbors: Number of nearest neighbors

    Returns:
        torch.Tensor: Mean shift loss.
    """
    p = F.normalize(p,dim=1)
    z = F.normalize(z,dim=1)
    queue = F.normalize(queue,dim=1)
    _,indices = torch.topk(z@queue.T,num_neighbors,dim=1)
    nn_targets = queue[indices]
    z = z.unsqueeze(dim=1)
    p = p.unsqueeze(dim=1)
    return 2 - 2*(
            torch.einsum('nik,njk->nj',p,nn_targets.detach())
            ).mean()


        

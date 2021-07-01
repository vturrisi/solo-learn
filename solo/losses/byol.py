import torch.nn.functional as F


def byol_loss_func(p, z, simplified=True):
    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()

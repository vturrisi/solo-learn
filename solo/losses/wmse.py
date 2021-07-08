import torch.nn.functional as F


def wmse_loss_func(z1, z2, simplified=True):
    if simplified:
        return 2 - 2 * F.cosine_similarity(z1, z2.detach(), dim=-1).mean()
    else:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

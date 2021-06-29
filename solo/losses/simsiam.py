import torch.nn.functional as F


def simsiam_loss_func(p, z, simplified=True):
    if simplified:
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return -(p * z.detach()).sum(dim=1).mean()

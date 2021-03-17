import torch.nn.functional as F


def negative_cosine_similarity(p, z):

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p * z.detach()).sum(dim=1).mean()

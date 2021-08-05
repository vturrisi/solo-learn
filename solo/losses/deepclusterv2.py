import torch.nn.functional as F


def deepclusterv2_loss_func(outputs, assignments, temperature=0.1):
    loss = 0
    for h in range(outputs.size(0)):
        scores = outputs[h].view(-1, outputs.size(-1)) / temperature
        targets = assignments[h].repeat(outputs.size(1)).to(outputs.device, non_blocking=True)
        loss += F.cross_entropy(scores, targets, ignore_index=-1)
    return loss / outputs.size(0)

from typing import Sequence

import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs: output of a classifier (logits or probabilities)
        targets: ground truth labels
        top_k: sequence of top k values to compute the accuracy over

    Returns:
        accuracies at the desired k

    """
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs, key, batch_size_key):
    """
    Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs: list of dicts containing the outputs of a validation step
        key: desired key
        batch_size_key: key of batch size values

    Returns:
        weighted mean of the values of a key

    """
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)

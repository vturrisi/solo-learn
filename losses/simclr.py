import torch
import torch.nn.functional as F


def simclr_loss_func(x1, x2, extra_pos_mask=None, temperature=0.2, normalize=True):
    assert x1.size() == x2.size()

    # get the current device based on x1
    device = x1.device

    b = x1.size(0)
    x = torch.cat((x1, x2), dim=0)
    if normalize:
        x = F.normalize(x, dim=1)

    logits = torch.einsum("if, jf -> ij", x, x) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)

    # if we have extra "positives"
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


def manual_simclr_loss_func(x, pos_mask, negative_mask, temperature=0.2, normalize=True):
    if normalize:
        x = F.normalize(x, dim=1)

    logits = torch.einsum("if, jf -> ij", x, x) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    negatives = torch.sum(torch.exp(logits) * negative_mask, dim=1, keepdim=True)
    exp_logits = torch.exp(logits)
    log_prob = torch.log(exp_logits / (exp_logits + negatives))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    indexes = pos_mask.sum(1) > 0
    pos_mask = pos_mask[indexes]
    mean_log_prob_pos = mean_log_prob_pos[indexes] / pos_mask.sum(1)

    # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss

import torch
from solo.losses import swav_loss_func
from solo.utils.sinkhorn_knopp import SinkhornKnopp


def get_assignments(preds):
    bs = preds[0].size(0)
    assignments = []
    sk = SinkhornKnopp(3, 0.05, 1)

    for i, p in enumerate(preds):
        # compute assignments with sinkhorn-knopp
        assignments.append(sk(p)[:bs])
    return assignments


def test_swav_loss():
    b, f = 32, 128
    preds = torch.randn(2, b, f).requires_grad_()
    assignments = get_assignments(preds)

    loss = swav_loss_func(preds, assignments, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = swav_loss_func(preds, assignments, temperature=0.1)
        loss.backward()
        preds.data.add_(-0.5 * preds.grad)

        preds.grad = None

    assert loss < initial_loss

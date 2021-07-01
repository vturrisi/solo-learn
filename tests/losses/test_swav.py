import torch
import torch.nn as nn
from solo.losses import swav_loss_func
from solo.utils.sinkhorn_knopp import SinkhornKnopp


def get_assignments(preds):
    bs = preds[0].size(0)
    assignments = []
    sk = SinkhornKnopp(10, 0.05, 1)

    for i, p in enumerate(preds):
        # compute assignments with sinkhorn-knopp
        assignments.append(sk(p)[:bs])
    return assignments


def test_swav_loss():
    b, f = 256, 128
    prototypes = nn.utils.weight_norm(torch.nn.Linear(f, f, bias=False))

    z = torch.zeros(2, b, f).uniform_(-2, 2).requires_grad_()
    preds = prototypes(z)
    assignments = get_assignments(preds)

    loss = swav_loss_func(preds, assignments, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        preds = prototypes(z)
        assignments = get_assignments(preds)
        loss = swav_loss_func(preds, assignments, temperature=0.1)
        loss.backward()

        z.data.add_(-0.5 * z.grad)
        z.grad = None

    assert loss < initial_loss

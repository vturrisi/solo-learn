import torch
from solo.losses import moco_loss_func


def test_moco_loss():
    b, f, q = 32, 128, 15000
    query = torch.randn(b, f).requires_grad_()
    key = torch.randn(b, f).requires_grad_()
    queue = torch.randn(f, q)

    loss = moco_loss_func(query, key, queue, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = moco_loss_func(query, key, queue, temperature=0.1)
        loss.backward()
        query.data.add_(-0.5 * query.grad)
        key.data.add_(-0.5 * key.grad)

        query.grad = key.grad = None

    assert loss < initial_loss

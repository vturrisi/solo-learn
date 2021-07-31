import torch
from solo.losses import ressl_loss_func


def test_moco_loss():
    b, f, q = 32, 128, 15000
    query = torch.randn(b, f).requires_grad_()
    key = torch.randn(b, f)
    queue = torch.randn(q, f)

    loss = ressl_loss_func(query, key, queue, temperature_q=0.1, temperature_k=0.04)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = ressl_loss_func(query, key, queue, temperature_q=0.1, temperature_k=0.04)
        loss.backward()
        query.data.add_(-0.5 * query.grad)

        query.grad = None

    assert loss < initial_loss

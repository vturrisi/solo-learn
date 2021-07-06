import torch
from solo.losses import wmse_loss_func


def test_simsiam_loss():
    b, f = 32, 128
    p = torch.randn(b, f).requires_grad_()
    z = torch.randn(b, f)

    loss = wmse_loss_func(p, z)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = wmse_loss_func(p, z)
        loss.backward()
        p.data.add_(-0.5 * p.grad)

        p.grad = None

    assert loss < initial_loss

    assert abs(wmse_loss_func(p, z) - wmse_loss_func(p, z, simplified=False)) < 1e-6

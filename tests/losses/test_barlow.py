import torch
from solo.losses import barlow_loss_func


def test_barlow_loss():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    loss = barlow_loss_func(z1, z2, lamb=5e-3, scale_loss=0.025)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = barlow_loss_func(z1, z2, lamb=5e-3, scale_loss=0.025)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss

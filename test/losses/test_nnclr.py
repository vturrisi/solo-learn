import torch
from solo.losses import nnclr_loss_func


def test_nnclr_loss():
    b, f = 32, 128
    nn = torch.randn(b, f)
    p = torch.randn(b, f).requires_grad_()

    loss = nnclr_loss_func(nn, p, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = nnclr_loss_func(nn, p, temperature=0.1)
        loss.backward()
        p.data.add_(-0.5 * p.grad)

        p.grad = None

    assert loss < initial_loss

import torch
from solo.losses import vicreg_loss_func


def test_vicreg_loss():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    loss = vicreg_loss_func(z1, z2, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = vicreg_loss_func(
            z1, z2, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0
        )
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss

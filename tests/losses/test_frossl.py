# Copyright 2024 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from solo.losses import frossl_loss_func


def test_frossl_loss_D_greaterthan_N():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    z = torch.stack([z1, z2], dim=0)
    loss = frossl_loss_func(z, invariance_weight=1.4)
    initial_loss = loss.item()
    assert initial_loss != 0

    for _ in range(20):
        z = torch.stack([z1, z2], dim=0)

        loss = frossl_loss_func(z, invariance_weight=1.4)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss



def test_frossl_loss_N_greaterthan_D():
    b, f = 128, 32
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    z = torch.stack([z1, z2], dim=0)
    loss = frossl_loss_func(z, invariance_weight=1.4)
    initial_loss = loss.item()
    assert initial_loss != 0

    for _ in range(20):
        z = torch.stack([z1, z2], dim=0)

        loss = frossl_loss_func(z, invariance_weight=1.4)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss

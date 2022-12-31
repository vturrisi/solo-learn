# Copyright 2023 solo-learn development team.

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
from solo.losses import barlow_loss_func


def test_barlow_loss():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    loss = barlow_loss_func(z1, z2, lamb=5e-3, scale_loss=0.025)
    initial_loss = loss.item()
    assert loss != 0

    for _ in range(20):
        loss = barlow_loss_func(z1, z2, lamb=5e-3, scale_loss=0.025)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss

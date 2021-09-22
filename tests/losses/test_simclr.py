# Copyright 2021 solo-learn development team.

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
from einops import repeat
from solo.losses import manual_simclr_loss_func, simclr_loss_func


def test_simclr_loss():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()

    loss = simclr_loss_func(z1, z2, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = simclr_loss_func(z1, z2, temperature=0.1)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss


def test_manual_simclr_loss():
    b, f = 32, 128
    n_augs = 2
    indexes = torch.arange(b)

    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()
    z = torch.cat((z1, z2))

    index_matrix = repeat(indexes, "b -> c (d b)", c=n_augs * indexes.size(0), d=n_augs)
    pos_mask = (index_matrix == index_matrix.t()).fill_diagonal_(False)
    neg_mask = (~pos_mask).fill_diagonal_(False)
    loss = manual_simclr_loss_func(z, pos_mask=pos_mask, neg_mask=neg_mask, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        z = torch.cat((z1, z2))
        loss = manual_simclr_loss_func(z, pos_mask=pos_mask, neg_mask=neg_mask, temperature=0.1)
        loss.backward()
        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss

    assert (
        abs(
            manual_simclr_loss_func(
                torch.cat((z1, z2)), pos_mask=pos_mask, neg_mask=neg_mask, temperature=0.1
            )
            - simclr_loss_func(z1, z2, temperature=0.1)
        )
        < 1e-6
    )

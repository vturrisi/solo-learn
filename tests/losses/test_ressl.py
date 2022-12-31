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
from solo.losses import ressl_loss_func


def test_moco_loss():
    b, f, q = 32, 128, 15000
    query = torch.randn(b, f).requires_grad_()
    key = torch.randn(b, f)
    queue = torch.randn(q, f)

    loss = ressl_loss_func(query, key, queue, temperature_q=0.1, temperature_k=0.04)
    initial_loss = loss.item()
    assert loss != 0

    for _ in range(20):
        loss = ressl_loss_func(query, key, queue, temperature_q=0.1, temperature_k=0.04)
        loss.backward()
        query.data.add_(-0.5 * query.grad)

        query.grad = None

    assert loss < initial_loss

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
import torch.nn as nn
from solo.losses import swav_loss_func
from solo.utils.sinkhorn_knopp import SinkhornKnopp


def get_assignments(preds):
    bs = preds[0].size(0)
    assignments = []
    sk = SinkhornKnopp(10, 0.05, 1)

    for p in preds:
        # compute assignments with sinkhorn-knopp
        assignments.append(sk(p)[:bs])
    return assignments


def test_swav_loss():
    b, f = 256, 128
    prototypes = nn.utils.weight_norm(torch.nn.Linear(f, f, bias=False))

    z = torch.zeros(2, b, f).uniform_(-2, 2).requires_grad_()
    preds = prototypes(z)
    assignments = get_assignments(preds)

    loss = swav_loss_func(preds, assignments, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for _ in range(20):
        preds = prototypes(z)
        assignments = get_assignments(preds)
        loss = swav_loss_func(preds, assignments, temperature=0.1)
        loss.backward()

        z.data.add_(-0.5 * z.grad)
        z.grad = None

    assert loss < initial_loss

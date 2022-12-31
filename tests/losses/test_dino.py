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
from solo.losses import DINOLoss


def test_dino_loss():
    b, f, num_epochs = 32, 128, 20
    p = torch.randn(b, f).requires_grad_()
    p_momentum = torch.randn(b, f)

    dino_loss = DINOLoss(
        num_prototypes=f,
        warmup_teacher_temp=0.4,
        teacher_temp=0.7,
        warmup_teacher_temp_epochs=10,
        student_temp=0.1,
        num_epochs=num_epochs,
    )

    loss = dino_loss(p, p_momentum)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        dino_loss.epoch = i

        loss = dino_loss(p, p_momentum)
        loss.backward()
        p.data.add_(-0.5 * p.grad)

        p.grad = None

    assert loss < initial_loss

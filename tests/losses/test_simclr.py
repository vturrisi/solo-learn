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
from solo.losses import simclr_loss_func


def test_simclr_loss():
    b, f = 32, 128
    z1 = torch.randn(b, f).requires_grad_()
    z2 = torch.randn(b, f).requires_grad_()
    z = torch.cat((z1, z2))
    indexes = torch.arange(b).repeat(2)

    loss = simclr_loss_func(z, indexes, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for _ in range(20):
        z = torch.cat((z1, z2))
        loss = simclr_loss_func(z, indexes, temperature=0.1)
        loss.backward()

        z1.data.add_(-0.5 * z1.grad)
        z2.data.add_(-0.5 * z2.grad)

        z1.grad = z2.grad = None

    assert loss < initial_loss


def test_simclr_loss_with_labels_is_supcon_l_out():
    # With class labels as indexes (the SupCon method), the loss is L_out^sup of Khosla et al. (2020), Eq. 2:
    # -mean_i 1/|P(i)| sum_{p in P(i)} log(exp(z_i.z_p / t) / sum_{a != i} exp(z_i.z_a / t)).
    torch.manual_seed(0)
    b, f, t = 32, 128, 0.2
    labels = torch.randint(0, 4, (b,)).repeat(2)
    z = torch.randn(2 * b, f)

    zn = torch.nn.functional.normalize(z, dim=-1)
    logits = zn @ zn.T / t
    self_mask = torch.eye(2 * b, dtype=torch.bool)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    log_norm = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
    log_prob = logits - log_norm
    expected = -((log_prob * pos_mask).sum(1) / pos_mask.sum(1)).mean()

    loss = simclr_loss_func(z, labels, temperature=t)
    assert torch.allclose(loss, expected, atol=1e-5)

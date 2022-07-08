# Copyright 2022 solo-learn development team.

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

import math

import torch
from torch import nn


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network.

    Args:
        online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
        momentum_net (nn.Module): momentum network (e.g. momentum backbone,
            momentum projection, etc...).
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_tau (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_tau (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        """Performs the momentum update for each param group.

        Args:
            online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
            momentum_net (nn.Module): momentum network (e.g. momentum backbone,
                momentum projection, etc...).
        """

        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )

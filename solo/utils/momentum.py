import math

import torch


@torch.no_grad()
def initialize_momentum_params(online_net, momentum_net):
    """
    Copies the parameters of the online network to the momentum network.

    Args:
        online_net: online network (e.g. online encoder, online projection, etc...)
        momentum_net: momentum network (e.g. momentum encoder, momentum projection, etc...)
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        """
        Updates momentum parameters using exponential moving average.

        Args:
            base_tau: base value of the weighting decrease coefficient (should be in [0,1])
            cur_tau: current value of the weighting decrease coefficient
            final_tau: current value of the weighting decrease coefficient (should be in [0,1])
        """

        super().__init__()
        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net, momentum_net):
        """
        Performs the momentum update for each param group.

        Args:
            online_net: online network (e.g. online encoder, online projection, etc...)
            momentum_net: momentum network (e.g. momentum encoder, momentum projection, etc...)
        """

        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step, max_steps):
        """
        Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step: number of gradient steps so far
            max_steps: overall number of gradient steps in the whole training

        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )

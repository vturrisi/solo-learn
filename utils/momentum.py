import math

import torch
from pytorch_lightning.callbacks import Callback


@torch.no_grad()
def initialize_momentum_params(online_encoder, momentum_encoder):
    params_online = online_encoder.parameters()
    params_momentum = momentum_encoder.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater():
    def __init__(self, base_tau=0.996, final_tau=1.0):
        super().__init__()
        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_nets, momentum_nets, cur_step, max_steps):

        # update weights
        for on, mn in zip(online_nets, momentum_nets):
            self.update_params(on, mn)

        # update tau afterward
        self.cur_tau = self.update_tau(cur_step, max_steps)

    def update_tau(self, cur_step, max_steps):
        tau = self.final_tau - (self.final_tau - self.base_tau) * \
              (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        return tau

    def update_params(self, online_nets, momentum_nets):
        for op, mp in zip(online_nets.parameters(), momentum_nets.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

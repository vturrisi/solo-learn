"""
References:
    - https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
"""
import torch
from torch.optim import Optimizer


class LARSWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        eta: float = 1e-3,
        clip: bool = False,
        eps: float = 1e-8,
        exclude_bias_n_norm: bool = False,
    ):
        """Wrapper that adds LARS scheduling to any optimizer.
        This helps stability with huge batch sizes.

        Args:
            optimizer (Optimizer): torch optimizer.
            eta (float, optional): trust coefficient. Defaults to 1e-3.
            clip (bool, optional): clip gradient values. Defaults to False.
            eps (float, optional): adaptive_lr stability coefficient. Defaults to 1e-8.
            exclude_bias_n_norm (bool, optional): exclude bias and normalization layers from lars.
                Defaults to False.
        """

        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

        self.__setstate__ = self.optim.__setstate__  # type: ignore
        self.__getstate__ = self.optim.__getstate__  # type: ignore
        self.__repr__ = self.optim.__repr__  # type: ignore

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property  # type: ignore
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, state):
        self.optim.state = state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None and (p.ndim != 1 or not self.exclude_bias_n_norm):
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr

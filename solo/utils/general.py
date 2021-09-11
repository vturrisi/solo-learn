from typing import List, Tuple

import torch
import torch.nn as nn


def _1d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite()


def _2d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite().all(dim=1)


def _single_input_filter(tensor: torch.Tensor) -> Tuple[torch.Tensor]:
    if len(tensor.size()) == 1:
        filter_func = _1d_filter
    elif len(tensor.size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensor)
    tensor = tensor[selected]

    return tensor, selected


def _multi_input_filter(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor]:
    if len(tensors[0].size()) == 1:
        filter_func = _1d_filter
    elif len(tensors[0].size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensors[0])
    for tensor in tensors[1:]:
        selected = torch.logical_and(selected, filter_func(tensor))
    tensors = [tensor[selected] for tensor in tensors]

    return tensors, selected


def filter_inf_n_nan(tensors: List[torch.Tensor], return_indexes: bool = False):
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (List): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """

    if isinstance(tensors, torch.Tensor):
        tensors, selected = _single_input_filter(tensors)
    else:
        tensors, selected = _multi_input_filter(tensors)

    if return_indexes:
        return tensors, selected
    return tensors


class FilterInfNNan(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise AttributeError()
            return getattr(self.module, name)

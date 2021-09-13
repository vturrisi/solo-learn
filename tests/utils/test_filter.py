import math

import torch
import torch.nn as nn
from solo.utils.misc import FilterInfNNan, filter_inf_n_nan


def test_filter():
    tensor = torch.randn(100)
    tensor[10] = math.nan
    filtered, selected = filter_inf_n_nan(tensor, return_indexes=True)
    assert filtered.size(0) == 99
    assert selected.sum() == 99

    tensor2 = torch.randn(100)
    assert [t.size(0) == 99 for t in filter_inf_n_nan([tensor, tensor2])]

    tensor = torch.randn(100, 30)
    tensor[10, 0] = math.inf
    assert filter_inf_n_nan(tensor).size(0) == 99

    tensor2 = torch.randn(100, 30)
    assert [t.size(0) == 99 for t in filter_inf_n_nan([tensor, tensor2])]

    class DummyNanLayer(nn.Module):
        def forward(self, x):
            x[10] = -math.inf
            return x

    dummy_nan_layer = DummyNanLayer()
    filter_layer = FilterInfNNan(dummy_nan_layer)
    assert filter_layer(tensor).size(0) == 99

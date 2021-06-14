import torch
from solo.utils.metrics import accuracy_at_k


def test_accuracy_at_k():
    b, c = 32, 100
    output = torch.randn(b, c)
    target = torch.randint(low=0, high=c, size=(b,))
    acc1, acc5 = accuracy_at_k(output, target)

    assert isinstance(acc1, torch.Tensor)
    assert isinstance(acc5, torch.Tensor)

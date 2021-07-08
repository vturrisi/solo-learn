import torch
from solo.utils.gather_layer import gather


def test_gather_layer():
    X = torch.randn(10, 30, requires_grad=True)
    X_gathered = gather(X)
    assert isinstance(X, torch.Tensor)

    dummy_loss = torch.mm(X_gathered, X_gathered.T).sum()
    dummy_loss.backward()
    assert X.grad is not None

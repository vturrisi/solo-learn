import torch


def filter_inf_n_nan(tensor: torch.Tensor) -> torch.Tensor:
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (torch.Tensor): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """
    tensor = tensor[~tensor.isnan()]
    tensor = tensor[~tensor.isinf()]
    return tensor

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gathers tensors from all processes, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [input]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            grad_out = torch.zeros_like(input)
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)

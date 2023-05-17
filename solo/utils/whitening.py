# Copyright 2023 solo-learn development team.

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


from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def __init__(self, output_dim: int, eps: float = 0.0):
        """Layer that computes hard whitening for W-MSE using the Cholesky decomposition.

        Args:
            output_dim (int): number of dimension of projected features.
            eps (float, optional): eps for numerical stability in Cholesky decomposition. Defaults
                to 0.0.
        """

        super().__init__()
        self.output_dim = output_dim
        self.eps = eps

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs whitening using the Cholesky decomposition.

        Args:
            x (torch.Tensor): a batch or slice of projected features.

        Returns:
            torch.Tensor: a batch or slice of whitened features.
        """

        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.output_dim, -1).mean(-1).view(1, -1, 1, 1)
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.output_dim, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.output_dim).type(f_cov.type())

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrinked), upper=False)[
            0
        ]
        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.output_dim, 1, 1)

        decorrelated = conv2d(xn, inv_sqrt)

        return decorrelated.squeeze(2).squeeze(2)


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> torch.Tensor:
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args

        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                beta=eps,
                input=P[0],
                alpha=1.0 / m,
                batch1=xc,
                batch2=xc.transpose(1, 2),
            )
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(
                    beta=1.5,
                    input=P[k],
                    alpha=-0.5,
                    batch1=torch.matrix_power(P[k], 3),
                    batch2=Sigma_N,
                )
            saved.extend(P)
            wm = P[ctx.T].mul_(
                rTr.sqrt()
            )  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}

            running_mean.copy_(momentum * mean + (1.0 - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1.0 - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        saved = ctx.saved_tensors
        if len(saved) == 0:
            return None, None, None, None, None, None, None, None

        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(beta=1.5, alpha=-0.5, batch1=g_tmp, batch2=P2)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P2, batch2=g_tmp)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1])
        g_sn += g_P
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum(
            (1, 2), keepdim=True
        ) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = (
            g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        )
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_groups: int = 64,
        num_channels: Optional[int] = None,
        T: int = 5,
        dim: int = 2,
        eps: float = 1.0e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        # assert dim == 4, 'IterNorm does not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert (
            num_groups > 0 and num_features % num_groups == 0
        ), f"num features={num_features}, num groups={num_groups}"
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*shape))
            self.bias = nn.Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer(
            "running_wm",
            torch.eye(num_channels).expand(num_groups, num_channels, num_channels).clone(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_hat = iterative_normalization_py.apply(
            X,
            self.running_mean,
            self.running_wm,
            self.num_channels,
            self.T,
            self.eps,
            self.momentum,
            self.training,
        )
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias

        return X_hat

    def extra_repr(self):
        return (
            f"{self.num_features}, num_channels={self.num_channels}, T={self.T}, eps={self.eps}, "
            "momentum={momentum}, affine={affine}"
        )

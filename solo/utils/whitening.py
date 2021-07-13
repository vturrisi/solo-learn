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

        super(Whitening2d, self).__init__()
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

        inv_sqrt = torch.triangular_solve(eye, torch.cholesky(f_cov_shrinked), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.output_dim, 1, 1)

        decorrelated = conv2d(xn, inv_sqrt)

        return decorrelated.squeeze(2).squeeze(2)

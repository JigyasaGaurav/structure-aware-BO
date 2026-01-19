import torch
import gpytorch
from gpytorch.kernels import Kernel
from config_loader import FINGERPRINT_NBITS
# from commons import print_kernel_debug_stats

def batch_tanimoto_sim(
    x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    from: https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py 
    Tanimoto similarity between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added. Tanimoto similarity is proportional to:

    :math:`(<x, y>) / (||x||^2 + ||y||^2 - <x, y>)`

    where x and y may be bit or count vectors or in set notation:

    :math:`|A \\cap B| / |A| + |B| - |A \\cap B|`

    Args:
        x1: `[b x n x d]` Tensor where b is the batch dimension
        x2: `[b x m x d]` Tensor
        eps: Float for numerical stability. Default value is 1e-6

    Returns:
        Tensor denoting the Tanimoto similarity.
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (
        eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )

    return tan_similarity.clamp_min_(
        0
    )  # zero out negative values for numerical stability


class AssignmentDistanceKernel(Kernel):
    """
    Computes a kernel as the product of:
    1. Assignment distance similarity between pairs of molecular fingerprints
    2. RBF kernel over Euclidean distance of ratios
    Each input is [fp1, fp2, n1, n2], where fp1 and fp2 are binary fingerprints.
    """
    def __init__(self, nBits=FINGERPRINT_NBITS, **kwargs):
        super().__init__(**kwargs)
        self.nBits = nBits

    def forward(self, x_a, x_b, **params):
        """
        Compute the kernel value between two sets of inputs.
        Args:
            x_a: Tensor of shape (N, D) for the first set of inputs.
            x_b: Tensor of shape (M, D) for the second set of inputs.
            params: Additional parameters (not used here).
        Returns:
            Tensor of shape (N, M) containing the kernel values.
        """
        fp1a = x_a[..., :self.nBits]
        fp2a = x_a[..., self.nBits:2*self.nBits]

        fp1b = x_b[..., :self.nBits]
        fp2b = x_b[..., self.nBits:2*self.nBits]

        s1 = batch_tanimoto_sim(fp1a, fp1b) + batch_tanimoto_sim(fp2a, fp2b)
        s2 = batch_tanimoto_sim(fp1a, fp2b) + batch_tanimoto_sim(fp2a, fp1b)
        return torch.max(s1/2, s2/2) 
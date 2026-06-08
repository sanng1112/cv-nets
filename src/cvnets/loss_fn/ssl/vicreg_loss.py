"""VICReg Loss for self-supervised learning.

References
----------
- Bardes, Adrien, Jean Ponce, and Yann LeCun. "VICReg: Variance-Invariance-
  Covariance Regularization for Self-Supervised Learning." ICLR 2022.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("vicreg_loss", category="ssl")
class VICRegLoss(BaseLoss):
    """VICReg loss with invariance, variance, and covariance terms.

    Parameters
    ----------
    sim_w : float
        Weight for invariance (MSE) term (default ``25.``).
    var_w : float
        Weight for variance (hinge) term (default ``25.``).
    cov_w : float
        Weight for covariance (off-diagonal) term (default ``1.``).
    eps : float
        Small constant for numerical stability in std computation (default ``1e-4``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        sim_w: float = 25.0,
        var_w: float = 25.0,
        cov_w: float = 1.0,
        eps: float = 1e-4,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.sim_w = sim_w
        self.var_w = var_w
        self.cov_w = cov_w
        self.eps = eps

    @staticmethod
    def _off_diagonal(x: Tensor) -> Tensor:
        """Return off-diagonal elements of the covariance matrix of *x*.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(N, D)``.

        Returns
        -------
        Tensor
            Flattened off-diagonal elements of the ``(D, D)`` covariance matrix.
        """
        n, d = x.shape
        # Center
        x_centered = x - x.mean(dim=0)
        # Covariance matrix (unbiased: divide by n-1)
        cov = (x_centered.T @ x_centered) / (n - 1)
        # Return flattened off-diagonal elements
        return cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute VICReg loss.

        Parameters
        ----------
        prediction : Tensor
            First view embeddings of shape ``(N, D)``.
        target : Tensor
            Second view embeddings of shape ``(N, D)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        # 1) Invariance term: MSE between prediction and target
        inv = F.mse_loss(prediction, target)

        # 2) Variance term: hinge on std to prevent collapse
        std1 = torch.sqrt(prediction.var(dim=0, unbiased=False) + self.eps)
        std2 = torch.sqrt(target.var(dim=0, unbiased=False) + self.eps)
        var = torch.relu(1.0 - std1).mean() + torch.relu(1.0 - std2).mean()

        # 3) Covariance term: penalise off-diagonal elements
        cov = (
            self._off_diagonal(prediction).pow(2).sum() / prediction.shape[1]
            + self._off_diagonal(target).pow(2).sum() / target.shape[1]
        )

        total = self.sim_w * inv + self.var_w * var + self.cov_w * cov
        return self._reduce(total.unsqueeze(0))

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, sim={self.sim_w}, "
            f"var={self.var_w}, cov={self.cov_w}"
        )

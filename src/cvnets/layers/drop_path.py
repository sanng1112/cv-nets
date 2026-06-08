"""DropPath / Stochastic Depth — randomly drops entire sample paths during training."""

from torch import Tensor, nn


class DropPath(nn.Module):
    """Stochastic Depth per sample (Huang et al., 2016).

    During training, each item in the batch is either kept (scaled by
    ``1 / (1 - drop_prob)``) or zeroed entirely with probability
    ``drop_prob``.  During evaluation this layer is a no-op.

    Parameters
    ----------
    drop_prob : float
        Probability of dropping a path (default ``0.0``, i.e. identity).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        if self.drop_prob == 1.0:
            return x.new_zeros(x.shape)
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + x.new_empty(shape).uniform_(0, 1)
        mask = random_tensor.floor_()
        return x.div(keep_prob) * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"

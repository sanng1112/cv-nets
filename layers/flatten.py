from typing import Optional

from torch import Tensor, nn


class Flatten(nn.Flatten):
    def __init__(self, start_dim: Optional[int] = 1, end_dim: Optional[int] = -1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)

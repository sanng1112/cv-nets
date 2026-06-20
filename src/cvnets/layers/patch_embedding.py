"""PatchEmbedding — convert an image to a sequence of patch tokens (ViT-style)."""

from typing import Tuple, Union

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project to embeddings.

    Uses a strided ``Conv2d`` with ``kernel_size == patch_size`` and
    ``stride == patch_size``, then flattens spatial dimensions and
    transposes to ``(B, N, C)`` token format.

    **Note**: This layer expects a 4D input tensor `(B, C, H, W)` and outputs
    a 3D tensor `(B, N, E)` where `N` is the number of patches and `E` is the
    embedding dimension.

    **Example Usage**:
    ```python
    import torch
    from cvnets.layers import PatchEmbedding

    layer = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(2, 3, 224, 224) # Batch size 2, 3 channels, 224x224
    out = layer(x)
    print(out.shape) # Output: torch.Size([2, 196, 768]) (196 = (224/16)*(224/16))
    ```

    Parameters
    ----------
    img_size : int or tuple of (int, int)
        Input image spatial size.
    patch_size : int or tuple of (int, int)
        Patch size (square or rectangular).
    in_channels : int
        Number of input channels.
    embed_dim : int
        Embedding dimension for each patch token.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"in_channels={self.in_channels}, embed_dim={self.embed_dim}, "
            f"num_patches={self.num_patches}"
        )

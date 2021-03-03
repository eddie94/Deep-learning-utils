import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768):
        """
          PatchEmbedding is a class for linear projection of flattened image patches
        it takes a (B, C, H, W) size image tensor as an input
        and returns a (B, (H/patch_size)*(W/patch_size) E) sized tensor

          for example if we input a (B, 3, 224, 224) sized tensor,
        the tensor is splitted into 16X16, which is 14X14 for each patch
        then the tensor is flattened and each 196 pixels have E dimensions of embedding

        :param in_channels: the input channels of the input image
        :param patch_size: the number of patches for spitting the image
        :param embedding_size: the embedded dimension for flattened patches
        """
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x):
        x = self.projection(x)

        return x

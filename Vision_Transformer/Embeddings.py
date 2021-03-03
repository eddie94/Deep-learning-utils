import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import repeat


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768, img_size: int = 224):
        """
          PatchEmbedding is a class for linear projection of flattened image patches
        it takes a (B, C, H, W) size image tensor as an input
        and returns a (B, (H/patch_size)*(W/patch_size)+1 E) sized tensor

          for example if we input a (B, 3, 224, 224) sized tensor,
        the tensor is splitted into 16X16, which is 14X14 for each patch
        then the tensor is flattened and each 196 pixels have E dimensions of embedding

          classification tokens are added after the projection is done
        tokens are learnable parameters of pytorch


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

        self.classification_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.positional_embedding = nn.Parameter(torch.randn((img_size//patch_size)**2+1, embedding_size))

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.projection(x)

        # add classification tokens
        classification_tokens = repeat(self.classification_token, "() n e -> b n e", b=batch_size)
        x = torch.cat([classification_tokens, x], dim=1)

        # add positional embeddings
        x += self.positional_embedding

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Reduce

from Vision_Transformer.Embeddings import PatchEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int = 768, num_heads: int = 8, dropout: float = 0):
        '''
          a module for multi head attention


        :param embedding_size: the size of the embedding
        :param num_heads: number of heads for multi head attention
        :param dropout: dropout rate
        '''
        super(MultiHeadAttention, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # query key value in one matrix
        self.qkv = nn.Linear(embedding_size, embedding_size*3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
          NOTE
        b = batch_size
        n = sequence length
        h = number of heads
        d = embedding size
        """
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # matrix multiplication between queries and keys
        summation = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            summation.masked_fill(~mask, fill_value)

        scaling = self.embedding_size**0.5
        attention = F.softmax(summation, dim=-1)/scaling
        attention = self.dropout(attention)

        # matrix multiplication between attention and values
        output = torch.einsum("bhal, bhlv -> bhav", attention, values)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.fc(output)

        return output


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4, dropout: float = 0.):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(embedding_size, expansion*embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion*embedding_size, embedding_size)
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 embedding_size: int = 768,
                 dropout: float = 0.,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.,
                 **kwargs):

        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                MultiHeadAttention(embedding_size, **kwargs),
                nn.Dropout(dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForwardBlock(
                    embedding_size, expansion=forward_expansion, dropout=forward_dropout
                ),
                nn.Dropout(dropout)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, embedding_size: int = 768, n_classes: int = 10):
        super(ClassificationHead, self).__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self, in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 10,
                 **kwargs):
        
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, embedding_size, img_size),
            TransformerEncoder(depth, embedding_size=embedding_size, **kwargs),
            ClassificationHead(embedding_size, n_classes)
        )


a = torch.zeros(1, 3, 224, 224)
b = ViT()(a)
print(b.shape)

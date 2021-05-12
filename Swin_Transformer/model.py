import torch
import torch.nn as nn

from einops import rearrange


class WindowAttention(nn.Module):
    def __init__(self, hidden_dim, window_size, num_heads):
        """
        computes the W-MSA
        input shape: (num_windows*B, N, C)
        where
        B: Batch_size
        N: number of patches
        C: hidden dimension for each patch

        :param hidden_dim: the hidden dimension of each for each patch
        :param window_size: the size of the window
        :param num_heads: number of head for MSA
        """
        super(WindowAttention, self).__init__()

        self.hidden_dim = hidden_dim
        if type(window_size) is int:
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size

        self.num_heads = num_heads
        head_dim = self.hidden_dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position representation part
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*self.window_size[0]-1) * (2*self.window_size[1]-1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[0] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init._no_grad_trunc_normal_(self.relative_position_bias_table, std=.02, mean=0., a=-2., b=2.)

        self.qkv = nn.Linear(hidden_dim, hidden_dim*3)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attention = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attention = attention + relative_position_bias.unsqueeze(0)
        attention = self.softmax(attention)

        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.fc(x)

        return x


a = WindowAttention(hidden_dim=256, window_size=4, num_heads=8)
a(torch.zeros(4, 16, 256))
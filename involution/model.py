import torch
import torch.nn as nn


class Inv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, reduction_ratio=1, groups=1,
                dilation=(1, 1), padding=(1, 1), **kwargs):
        super(Inv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.feature_extractor = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                           bias=False, padding=(0, 0))\
            if in_channels != out_channels else nn.Identity()

        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.o = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//reduction_ratio, kernel_size=1,
                                bias=False)
        self.span = nn.Conv2d(in_channels=out_channels//reduction_ratio,
                              out_channels=self.kernel_size[0]*self.kernel_size[1]*groups, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels//reduction_ratio)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # check the input dimension
        assert x.ndimension() == 4

        batch_size, in_channels, height, width = x.shape

        unfold = self.feature_extractor(x)
        unfold = self.unfold(unfold)
        unfold = unfold.view(batch_size, self.groups, self.out_channels//self.groups,
                             self.kernel_size[0]*self.kernel_size[1], height, width)

        # kernel generation
        kernel = self.span(self.reduce(self.o(x)))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0]*self.kernel_size[1], height, width
        ).unsqueeze(dim=2)

        output = (kernel * unfold).sum(dim=3).view(batch_size, -1, height, width)

        return output

"""Toy CNN example similar to AlexNet.

This version avoids ``nn.Conv2d``/``nn.MaxPool2d`` as well as
``torch.nn.functional.conv2d`` and ``torch.nn.functional.max_pool2d``.
``ManualConv2d`` and ``ManualMaxPool2d`` implement these operations using
``torch.nn.Unfold``.
"""

import torch
from torch import nn


class ManualConv2d(nn.Module):
    """A convolution layer implemented with :class:`torch.nn.Unfold`."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        # Convert image blocks into columns
        cols = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # cols: [N, C_in * k*k, L] where L is output spatial locations
        weight = self.weight.view(self.out_channels, -1)
        # Compute convolution via matrix multiplication
        # (N, L, C_in*k*k) @ (C_in*k*k, C_out) -> (N, L, C_out)
        out = cols.transpose(1, 2) @ weight.t()
        out = out.transpose(1, 2)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.view(n, self.out_channels, out_h, out_w)
        return out


class ManualMaxPool2d(nn.Module):
    """A max pooling layer implemented via :class:`torch.nn.Unfold`."""

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        cols = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        cols = cols.view(n, c, self.kernel_size * self.kernel_size, -1)
        out, _ = cols.max(dim=2)
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.view(n, c, out_h, out_w)
        return out

class ToyAlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ManualConv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            ManualMaxPool2d(kernel_size=3, stride=2),
            ManualConv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            ManualMaxPool2d(kernel_size=3, stride=2),
            ManualConv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ManualConv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ManualConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ManualMaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, height, width]
        x = self.features(x)  # -> [batch, 256, 6, 6]
        x = torch.flatten(x, 1)  # -> [batch, 256*6*6]
        x = self.classifier(x)  # -> [batch, num_classes]
        return x


def demo():
    # Create a ToyAlexNet and run a forward pass with random data.
    net = ToyAlexNet(num_classes=10)
    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 images
    out = net(dummy)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    demo()

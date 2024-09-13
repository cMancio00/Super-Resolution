import torch
from torch import nn
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += res
        return out


class Upsample(nn.Module):

    def __init__(self, num_channel):
        super().__init__()
        self.conv = Conv2d(num_channel, num_channel * 4, kernel_size=3, padding=1)
        self.shuffle = PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.k = 1/(in_channels * np.power(kernel_size, 2))

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels).uniform_(-np.sqrt(self.k),np.sqrt(self.k)))
        else:
            self.bias = 0

        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size).uniform_(-np.sqrt(self.k),np.sqrt(self.k)))

    def forward(self, image):
        image = nn.functional.pad(image, (self.padding,) * 4, "constant", 0)
        batch_size, in_channels, height, width = image.shape
        out_channels, in_channels_kernel, m, n = self.weight.shape
        if self.in_channels != in_channels:
            raise ValueError(
                f"Input channels are different: Declared {self.in_channels}, but got Image with {in_channels}")
        output_height = height - m + 1
        output_width = width - n + 1
        new_image = torch.zeros((batch_size, out_channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        new_image[b, c, i, j] = torch.sum(image[b, :, i:i + m, j:j + n] * self.weight[c]) + self.bias[c]
        return new_image


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        return torch.reshape(x,
                             (batch_size, channels / (self.upscale_factor**2),
                              height*self.upscale_factor, width*self.upscale_factor))

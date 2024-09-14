import torch
from torch import nn
import numpy as np


def _im2col(input, kernel_size, stride=1, padding=0):
    input_padded = torch.nn.functional.pad(input, (padding, padding, padding, padding))
    batch_size, in_channels, height, width = input_padded.size()
    kernel_height, kernel_width = kernel_size
    out_height = (height - kernel_height) // stride + 1
    out_width = (width - kernel_width) // stride + 1
    col = torch.empty(batch_size, in_channels, kernel_height, kernel_width, out_height, out_width)

    for y in range(kernel_height):
        for x in range(kernel_width):
            col[:, :, y, x, :, :] = input_padded[:, :, y: y + out_height * stride: stride,
                                    x: x + out_width * stride: stride]

    return col.view(batch_size, in_channels * kernel_height * kernel_width, -1)

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
            self.bias = torch.zeros(out_channels)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size).uniform_(-np.sqrt(self.k),np.sqrt(self.k)))


    def _conv_forward(self, input, weight, bias=None, stride=1, padding=0):
        col = _im2col(input, weight.size()[2:], stride, padding)
        # (out_channels, in_channels * kernel_height * kernel_width)
        weight_col = weight.view(weight.size(0), -1)
        out = torch.matmul(weight_col, col)

        if bias is not None:
            out += bias.view(1, -1, 1)

        batch_size, out_channels = out.size(0), weight.size(0)
        out_height = (input.size(2) + 2 * padding - weight.size(2)) // stride + 1
        out_width = (input.size(3) + 2 * padding - weight.size(3)) // stride + 1
        return out.view(batch_size, out_channels, out_height, out_width)

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias, padding=self.padding)


    def slow_forward(self, image):
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
        channels //= (self.upscale_factor ** 2)
        x = x.view(batch_size, channels, self.upscale_factor, self.upscale_factor, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.contiguous().view(batch_size, channels, height * self.upscale_factor, width * self.upscale_factor)

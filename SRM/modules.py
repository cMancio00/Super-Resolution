import torch
from torch import nn


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
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, image: torch.Tensor):
        pad = nn.ReplicationPad2d((self.padding,)*4)
        padded_image = pad(image)
        unfolded_image = torch.nn.functional.unfold(padded_image, kernel_size=self.kernel_size, stride=self.stride)
        kernel_reshaped = self.kernel.view(self.out_channels, -1)
        unfolded_output = torch.matmul(kernel_reshaped, unfolded_image)
        output_height = (image.size(2) + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (image.size(3) + 2 * self.padding - self.kernel_size) // self.stride + 1
        return unfolded_output.view(image.size(0), self.out_channels, output_height, output_width)









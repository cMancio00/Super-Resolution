from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

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
        self.conv = nn.Conv2d(num_channel, num_channel * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out


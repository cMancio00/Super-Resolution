from SRM.modules import *


class SuperResolution(nn.Module):

    def __init__(self, num_channels: int, num_res_block: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        ])
        for _ in range(num_res_block):
            self.layers.append(ResidualBlock(num_channels))
            self.layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
        self.layers.append(Upsample(num_channels))
        self.layers.append(nn.Conv2d(num_channels * 4, 3, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

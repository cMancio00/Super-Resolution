import numpy
from tqdm import tqdm
from SRM.modules import *
import torch


class SuperResolution(nn.Module):

    def __init__(self, num_channels: int, num_res_block: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        ])
        for _ in range(num_res_block):
            self.layers.append(ResidualBlock(num_channels))
        self.layers.append(Upsample(num_channels))
        self.layers.append(nn.Conv2d(num_channels, 3, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_loop(self, loss_fn, optimiser, epochs, train_dataloader, device='cpu') -> numpy.ndarray:
        losses = torch.zeros(epochs, device=device)
        print(f"Running on {device}")
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for low_res, high_res in train_dataloader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                optimiser.zero_grad()
                output = self(low_res)
                loss = loss_fn(output, high_res)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{epochs} average L1 Loss: {avg_loss:.6f}')
            losses[epoch] = avg_loss
        return losses.cpu().numpy()

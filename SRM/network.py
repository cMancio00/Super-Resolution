from typing import Any
from SRM.modules import *
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio


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

    def training_loop(self, loss_fn, optimiser, epochs, train_dataloader, device='cpu') -> tuple[Any, Any]:
        losses = torch.zeros(epochs, device=device)
        psnr = torch.zeros(epochs, device=device)
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_psnr = 0
            for low_res, high_res in train_dataloader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                optimiser.zero_grad()
                output = self(low_res)
                loss = loss_fn(output, high_res)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                epoch_psnr += peak_signal_noise_ratio(output.detach(), high_res)

            avg_loss = epoch_loss / len(train_dataloader)
            avg_psnr = epoch_psnr / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{epochs}: Average L1 Loss: {avg_loss:.6f}, Average PSNR: {avg_psnr:.6f} db')
            losses[epoch] = avg_loss
            psnr[epoch] = avg_psnr
        return losses.cpu().numpy(), psnr.cpu().numpy()

    def test(self, loss_fn, test_dataloader, device='cpu'):
        self.to(device)
        self.eval()
        total_loss = 0.
        psnr = 0.
        for low_res, high_res in test_dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            with torch.no_grad():
                predicted_high_res = self(low_res)
                loss = loss_fn(predicted_high_res, high_res)
                total_loss += loss.item()
                psnr += peak_signal_noise_ratio(predicted_high_res, high_res)

        avg_loss = total_loss / len(test_dataloader)
        avg_psnr = psnr / len(test_dataloader)
        return avg_loss, avg_psnr

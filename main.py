import os
import torch
from torch.utils.data import DataLoader
from dataset.data_preparation import download
from dataset.super_resolution_dataset import SuperResolutionDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    download("./data", "airplanes")
    root_dir = 'data/airplanes'
    dataset = SuperResolutionDataset(root_dir=root_dir, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    save_dir = 'data/preprocessed_dataset'
    os.makedirs(save_dir, exist_ok=True)

    for i, (low_res, high_res) in enumerate(train_dataloader):
        torch.save(low_res, os.path.join(save_dir, f'low_res_{i}.pt'))
        torch.save(high_res, os.path.join(save_dir, f'high_res_{i}.pt'))

    low_res, high_res = next(iter(train_dataloader))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Tensori (canali, altezza, larghezza)
    # matplot (altezza, larghezza, canali)
    ax[0].imshow(low_res[0].permute(1, 2, 0))
    ax[0].set_title("Low Resolution")

    ax[1].imshow(high_res[0].permute(1, 2, 0))
    ax[1].set_title("High Resolution")

    plt.show()


if __name__ == "__main__":
    main()

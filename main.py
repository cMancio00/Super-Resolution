import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.data_preparation import download, split_dataset
from dataset.super_resolution_dataset import SuperResolutionDataset
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from SRM.network import SuperResolution
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sizes = {
    "train": 0.5,
    "validation": 0.3,
    "test": 0.2
}
model_parameters = {
    "num_channels": 64,
    "num_res_block": 16
}
SRN = SuperResolution(**model_parameters)
hyperparameters = {
    "params": SRN.parameters(),
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8
}


def main():
    torch.manual_seed(777)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    download("./data", "airplanes")
    root_dir = 'data/airplanes'
    dataset = SuperResolutionDataset(root_dir=root_dir, transform=transform)

    train, validation, test = split_dataset(dataset, sizes)

    train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=16, shuffle=True)

    loss_fn = nn.L1Loss()

    optimiser = optim.Adam(**hyperparameters)
    training_parameters = {
        "loss_fn": loss_fn,
        "optimiser": optimiser,
        "epochs": 250,
        "train_dataloader": train_dataloader,
        "device": device
    }
    training_start = time.time()
    losses = SRN.training_loop(**training_parameters)
    training_end = time.time()
    total_time = training_end - training_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds.")

    os.makedirs("training_loss", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    loss_name = f"training_loss/L1_{timestamp}.csv"
    np.savetxt(loss_name, losses, delimiter=",")

    os.makedirs("checkpoint", exist_ok=True)

    model_filename = \
        f"checkpoint/SR_c{model_parameters["num_channels"]}_" + \
        f"rb{model_parameters["num_res_block"]}_" + \
        f"e{training_parameters["epochs"]}_{timestamp}.pth"
    torch.save(SRN.state_dict(), model_filename)
    print(f'Model saved as {model_filename}')


if __name__ == "__main__":
    main()

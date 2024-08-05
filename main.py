from torch.utils.data import DataLoader
from dataset.data_preparation import download, split_dataset
from dataset.super_resolution_dataset import SuperResolutionDataset
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from utils.training_utilitis import *

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
        "epochs": 2,
        "train_dataloader": train_dataloader,
        "device": device
    }
    training_start = time.time()
    losses, psnr = SRN.training_loop(**training_parameters)
    training_end = time.time()
    print(format_training_time(training_end - training_start))

    save_training_logs(losses, psnr)
    save_checkpoint(SRN, model_parameters, training_parameters)


if __name__ == "__main__":
    main()

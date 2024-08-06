from torch.utils.data import DataLoader
from dataset.data_preparation import download, split_dataset
from dataset.super_resolution_dataset import SuperResolutionDataset
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import time
from utils.training_utilitis import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define sizes for data splitting
sizes = {
    "train": 0.5,
    "validation": 0.3,
    "test": 0.2
}

# Define validation parameters to choose from
validation_parameters = {
    "num_channels": [16, 32, 64],
    "num_res_block": [4, 8, 16]
}

training_epochs = 1
final_training_epochs = 2


def main():
    print(f"Running on {device}")
    torch.manual_seed(777)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Download and extract the dataset
    download("./data", "airplanes")
    root_dir = 'data/airplanes'
    dataset = SuperResolutionDataset(root_dir=root_dir, transform=transform)

    # Split the dataset and make the final training dataset (to be used after model selection)
    train, validation, test = split_dataset(dataset, sizes)
    train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=16, shuffle=True)
    final_training_dataloader = DataLoader(
        torch.utils.data.ConcatDataset(
            [train, validation]
        ), batch_size=16, shuffle=True
    )

    # Model selection
    best_parameters, checkpoint_path = model_selection(
        train_dataloader,
        training_epochs,
        validation_dataloader,
        validation_parameters,
        device
    )
    print(f"Model selection is completed!")

    # Defining final Training model
    # Train the best model from checkpoint with train+validation dataset
    best_model = SuperResolution(**best_parameters)
    print(f"Loading checkpoint {checkpoint_path}...")
    best_model.load_state_dict(torch.load(checkpoint_path))

    training_parameters = generate_training_parameters(
        best_model, final_training_dataloader, final_training_epochs, device
    )

    training_start = time.time()
    losses, psnr = best_model.training_loop(**training_parameters)
    training_end = time.time()
    print(format_training_time(training_end - training_start))

    save_training_logs(losses, psnr)
    save_checkpoint(best_model, best_parameters, training_parameters)

    # Model Assessment
    avg_loss, avg_psnr = best_model.test(nn.L1Loss(), test_dataloader, device)
    print(f"Test L1: {avg_loss}, PSNR {avg_psnr} db")


if __name__ == "__main__":
    main()

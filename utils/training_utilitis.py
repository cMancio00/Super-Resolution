import os
from datetime import datetime
from typing import Any, Dict
import numpy as np
import torch
from torch import nn, optim
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio
import SRM.network
from SRM.network import SuperResolution
from itertools import product


def format_training_time(total_time):
    """
    This function format a string with hours, minutes and seconds, given a time in seconds

    Args:
        total_time: total time in seconds

    Returns: A string with formated hours, minutes and seconds

    """
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    return f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds."


def save_training_logs(losses, psnr) -> None:
    """
    Saves training logs in the folder training_logs
    Args:
        losses: loss array
        psnr: psnr array

    Returns: None

    """
    os.makedirs("training_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    loss_name = f"training_logs/{timestamp}_L1.csv"
    psnr_name = f"training_logs/{timestamp}_psnr.csv"
    np.savetxt(loss_name, losses, delimiter=",")
    np.savetxt(psnr_name, psnr, delimiter=",")
    print("Logs saved in training_logs")


def save_checkpoint(model: SuperResolution, model_parameters: dict, training_parameters: dict) -> str:
    """
    Saves the checkpoint of a given model after training in the folder checkpoint
    Args:
        model: Model to save
        model_parameters: dictionary of the model parameters
        training_parameters: dictionary of the training parameters

    Returns: path to checkpoint

    """
    os.makedirs("checkpoint", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_filename = \
        f"checkpoint/SR_c{model_parameters["num_channels"]}_" + \
        f"rb{model_parameters["num_res_block"]}_" + \
        f"e{training_parameters["epochs"]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved in {model_filename}')
    return model_filename


def generate_parameters(num_channels: list[int], num_res_block: list[int]) -> list[dict[str, Any]]:
    """
    Generate all possible combination of parameters for the model selection.

    Args:
        num_channels: list of possible num_channels
        num_res_block: list of possible number of residual blocks

    Returns: list of dictionary with model parameters

    """
    combinations = product(num_channels, num_res_block)
    return [{"num_channels": num_channels, "num_res_block": num_res_block} for
            num_channels, num_res_block in combinations]


def validate(
    model: SRM.network.SuperResolution,
    validation_dataloader: DataLoader,
    training_parameters: dict
) -> (float, float):
    """
    Validate a specific model during model selection
    Args:
        model: model to be validated
        validation_dataloader: validation set
        training_parameters: training parameters of the model

    Returns: Average Loss and PSNR found during validation

    """
    device = training_parameters["device"]
    loss_fn = training_parameters["loss_fn"]
    model = model.to(device)
    model.eval()
    total_loss = 0.
    total_psnr = 0.
    for low_res, high_res in validation_dataloader:
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        with torch.no_grad():
            predicted_high_res = model(low_res)
            loss = loss_fn(predicted_high_res, high_res)

            total_loss += loss.item()
            total_psnr += peak_signal_noise_ratio(predicted_high_res, high_res)

    avg_loss = total_loss / len(validation_dataloader)
    avg_psnr = total_psnr / len(validation_dataloader)

    return avg_loss, avg_psnr


def model_selection(
        train_dataloader: DataLoader,
        training_epochs: int,
        validation_dataloader: DataLoader,
        validation_parameters: dict[str, list[int]],
        device: str
) -> tuple[dict[str, Any], str]:
    """
    Select the best model given a set of parameters.
    Delegates the training of each Super Resolution model to its training method
    Args:
        train_dataloader: training set
        training_epochs: number of epochs for training
        validation_dataloader: validation set
        validation_parameters: dictionary with model parameters to validate
        device: device for computing training

    Returns: best model parameters found and the checkpoint path to the trained model

    """
    # Generate all possible combination of parameters to validate (channels,resBlock)
    parameters_combinations = generate_parameters(**validation_parameters)

    # Model selection loop
    best_loss = float('inf')
    for model_parameter in parameters_combinations:
        print(f"num_channels:{model_parameter["num_channels"]}, num_res_block:{model_parameter["num_res_block"]}")
        # Create and train a specific model
        SRN = SuperResolution(**model_parameter)
        training_parameters = generate_training_parameters(SRN, train_dataloader, training_epochs, device)
        training_loss, training_psnr = SRN.training_loop(**training_parameters)
        # Validate the model
        avg_loss, avg_psnr = validate(SRN, validation_dataloader, training_parameters)
        print(f"{avg_loss}, {avg_psnr} db")
        # Save the current best loss and other metrics
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_psnr = avg_psnr
            best_model = SRN
            best_model_parameters = model_parameter
            best_training_loss = training_loss
            best_training_psnr = training_psnr
    # Print the best model found
    print(f"Best model has num_channels:{best_model_parameters["num_channels"]}, " +
          f"num_res_block:{best_model_parameters["num_res_block"]}\n" +
          f"Got L1: {best_loss}, {best_psnr} db in validation")
    # Save best training logs and checkpoint
    save_training_logs(best_training_loss, best_training_psnr)
    checkpoint_path = save_checkpoint(best_model, best_model_parameters, training_parameters)
    return best_model_parameters, checkpoint_path


def generate_training_parameters(model: nn.Module, train_dataloader: DataLoader, training_epochs: int, device: str)\
        -> dict[str, L1Loss | Adam | Any]:
    """
    Generate the dictionary used for training. It does not train the model.
    Args:
        model: A model to be trained
        train_dataloader: dataset to train the model
        training_epochs: number of epochs for training
        device: device for computing training

    Returns: The dictionary with parameters to be passed to the training method

    """
    # Default hyperparameters of the paper (also ADAM defaults in pytorch)
    hyperparameters = {
        "params": model.parameters(),
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    loss_fn = nn.L1Loss()
    optimiser = optim.Adam(**hyperparameters)
    training_parameters = {
        "loss_fn": loss_fn,
        "optimiser": optimiser,
        "epochs": training_epochs,
        "train_dataloader": train_dataloader,
        "device": device
    }
    return training_parameters

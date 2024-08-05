import os
from datetime import datetime
import numpy as np
import torch
from SRM.network import SuperResolution


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


def save_checkpoint(model: SuperResolution, model_parameters: dict, training_parameters: dict) -> None:
    """
    Saves the checkpoint of a given model after training in the folder checkpoint
    Args:
        model: Model to save
        model_parameters: dictionary of the model parameters
        training_parameters: dictionary of the training parameters

    Returns:

    """
    os.makedirs("checkpoint", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_filename = \
        f"checkpoint/SR_c{model_parameters["num_channels"]}_" + \
        f"rb{model_parameters["num_res_block"]}_" + \
        f"e{training_parameters["epochs"]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved in {model_filename}')

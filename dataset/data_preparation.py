import torch as th
from torchvision.datasets import Caltech101
from shutil import move, rmtree
import os


def download(path : str, dataset : str):
    """Download a specific dataset in a given folder

    Args:
        path (str): folder to download the dataset
        dataset (str): dataset name

    Raises:
        ValueError: The dataset is not a valid option
    """
    if dataset not in ['airplanes']:
        raise ValueError(f"Invalid dataset choice: {dataset}")

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.listdir(path):
        Caltech101(path,target_type="category",download=True)
        move(path + "/caltech101/101_ObjectCategories/" + dataset, path)
        rmtree(path + "/caltech101")
    else:
        print(f"Dataset {dataset} already exists.")


if __name__ == "__main__":
    download("./data", "airplanes")


import torch
from torchvision.datasets import Caltech101
from shutil import move, rmtree
import os
from torch.utils.data import Dataset, Subset


def download(path: str, dataset: str):
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
        Caltech101(path, target_type="category", download=True)
        move(path + "/caltech101/101_ObjectCategories/" + dataset, path)
        rmtree(path + "/caltech101")
    else:
        print(f"Dataset {dataset} already exists, skipping download.")


def _random_split(dataset: Dataset, lengths: list[int]) -> list[Subset]:
    """
    Utility method that simulate pytorch random split
    Args:
        dataset: Dataset to be split
        lengths: List of lengths to split the dataset into

    Returns:
        List of Subsets
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(len(dataset)).tolist()
    subsets = []
    offset = 0
    for length in lengths:
        subset = torch.utils.data.Subset(dataset, indices[offset:offset + length])
        subsets.append(subset)
        offset += length
    return subsets

def split_dataset(dataset: Dataset, sizes: dict[str, float]) -> list[Subset]:
    """
        Splits a dataset into training, validation, and test sets based on the provided sizes.

        Args:
            dataset: The dataset to be split.
            sizes: A dictionary containing the sizes of the training, validation, and test sets.
                The keys must be 'train', 'validation', and 'test', and the values must sum up to 1.0.

        Returns:
            A list of three Subset objects, representing the training, validation, and test sets.

        Raises:
            ValueError: If the sizes dictionary does not contain the required keys or if the sizes do not sum up to 1.0.
        """
    required_keys = {'train', 'validation', 'test'}
    if set(required_keys) != set(sizes.keys()):
        raise ValueError(f"Dictionary of sizes must contain 'train', 'validation', and 'test' keys")
    if not sum(sizes.values()) == 1.0:
        raise ValueError(f"Sizes do not summ up to 1.0,but got {sum(sizes.values()):.2f}")
    train_size = int(sizes["train"] * len(dataset))
    validation_size = int(sizes["validation"] * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    return _random_split(dataset, [train_size, validation_size, test_size])

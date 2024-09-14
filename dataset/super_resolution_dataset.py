import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir, low_resolution=(128, 64), high_resolution=(256, 128),
                 transform=transforms.Compose([transforms.ToTensor()])) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = os.listdir(root_dir)
        self.low_resolution = low_resolution
        self.high_resolution = high_resolution

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the image in low and high resolution and convert them in tensor form
        Args:
            idx: index of the dataset

        Returns: tuple of low resolution image and high resolution image, in tensor form

        """
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_path)
        low_res = image.resize(self.low_resolution)
        high_res = image.resize(self.high_resolution)

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res

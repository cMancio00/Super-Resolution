import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([transforms.ToTensor()])):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_path)
        low_res = image.resize((128, 64))
        high_res = image.resize((256, 128))

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res

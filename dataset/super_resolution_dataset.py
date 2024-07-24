import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = os.listdir(root_dir)
        self.split = split

        random.seed(42)
        random.shuffle(self.file_names)
        train_size = int(0.8 * len(self.file_names))
        val_size = int(0.1 * len(self.file_names))
        self.train_files = self.file_names[:train_size]
        self.val_files = self.file_names[train_size:train_size+val_size]
        self.test_files = self.file_names[train_size+val_size:]

    def __len__(self):
        if self.split == 'train':
            return len(self.train_files)
        elif self.split == 'val':
            return len(self.val_files)
        else:
            return len(self.test_files)

    def __getitem__(self, idx):
        if self.split == 'train':
            img_path = os.path.join(self.root_dir, self.train_files[idx])
        elif self.split == 'val':
            img_path = os.path.join(self.root_dir, self.val_files[idx])
        else:
            img_path = os.path.join(self.root_dir, self.test_files[idx])

        image = Image.open(img_path)

        low_res = image.resize((128, 64), resample=Image.BICUBIC)

        high_res = image.resize((256, 128), resample=Image.BICUBIC)

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res

transform = transforms.Compose([
    transforms.ToTensor(),
])

root_dir = './data/airplanes/'
dataset = SuperResolutionDataset(root_dir=root_dir)

train_dataloader = DataLoader(SuperResolutionDataset(root_dir=root_dir, transform=transform, split='train'), batch_size=32, shuffle=True)
val_dataloader = DataLoader(SuperResolutionDataset(root_dir=root_dir, transform=transform, split='val'), batch_size=32, shuffle=False)
test_dataloader = DataLoader(SuperResolutionDataset(root_dir=root_dir, transform=transform, split='test'), batch_size=1, shuffle=False)

low_res, high_res = next(iter(train_dataloader))

plt.figure(figsize=(8, 8))
grid = make_grid(low_res[:4], nrow=2)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title('Low Resolution Images')
plt.show()

plt.figure(figsize=(8, 8))
grid = make_grid(high_res[:4], nrow=2)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title('High Resolution Images')
plt.show()
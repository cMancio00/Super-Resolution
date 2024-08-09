import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from SRM.network import SuperResolution
from dataset.data_preparation import download, split_dataset
from dataset.super_resolution_dataset import SuperResolutionDataset
from utils.training_utilitis import validate

# NOTE:
# Following tests are useless, they are only used in CI to verify that
# validation and test are executed without errors.
# There is NO assertions


class TestProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(777)
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.download_path = "./data"
        cls.category = "airplanes"
        download(cls.download_path, cls.category)
        root_dir = 'data/airplanes'
        dataset = SuperResolutionDataset(root_dir=root_dir)
        sizes = {
            "train": 0.5,
            "validation": 0.3,
            "test": 0.2
        }
        _, cls.validation, cls.test = split_dataset(dataset, sizes)

        cls.validation_dataloader = DataLoader(cls.test, batch_size=16, shuffle=True)
        cls.test_dataloader = DataLoader(cls.test, batch_size=16, shuffle=True)

    def test_model_selection(self, validation_dataloader=None):
        model_filename = "checkpoint/SR_c64_rb8_e50_202408051714.pth"
        validation_SRN = SuperResolution(64, 8)
        checkpoint_path = model_filename
        validation_SRN.load_state_dict(
            torch.load(checkpoint_path, map_location=self.__class__.device)
        )

        validate(validation_SRN, self.validation_dataloader,
                              {"loss_fn": nn.L1Loss(), "device": self.__class__.device})

    def test_model_assesment(self):
        model_filename = "checkpoint/SR_c64_rb8_e150_202408051740.pth"
        SRN = SuperResolution(64, 8)
        checkpoint_path = model_filename
        SRN.load_state_dict(
            torch.load(checkpoint_path, map_location=self.__class__.device)
        )

        validate(SRN, self.test_dataloader,
                 {"loss_fn": nn.L1Loss(), "device": self.__class__.device})


if __name__ == '__main__':
    unittest.main()

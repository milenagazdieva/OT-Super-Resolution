from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets
import random
from utils.utils import *
import numpy as np

class AugDataset(Dataset):
    """ Augmented dataset via random crops, flips and rotations.
    
    Parameters
    ----------
    datadir : str
        Path to images dataset.
    crop_size : int
        Size of crops from input images.
    flips : bool
        Augment by random flips or not.
    rotations : bool
        Augment by random rotations or not.
        
    Returns
    -------
    image : torch.Tensor
        Normalized image of crop_size from augmented dataset.
    """
    def __init__(self, datadir, crop_size, flips=False, rotations=False, **kwargs):
        super(AugDataset, self).__init__()
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size)
        ])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.ImageFolder(datadir, transform=self.input_transform)
        self.rotations = rotations
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
        return self.normalize(image)


class TestDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale_factor, crop_size=None, **kwargs):
        super(TestDataset, self).__init__()
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.hr_dataset = datasets.ImageFolder(hr_dir)
        self.lr_dataset = datasets.ImageFolder(lr_dir)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.hr_dataset)      
    
    def __getitem__(self, index):
        hr_image, _ = self.hr_dataset[index]
        lr_image, _ = self.lr_dataset[index]
        w, h = hr_image.size
        if self.crop_size == None:
            cs = calculate_valid_crop_size((h, w), self.scale_factor)
        else:
            cs = self.crop_size

        hr_image = T.CenterCrop(cs)(hr_image)
        hr_image = self.normalize(hr_image)
        if isinstance(cs, int):
            lr_cs =  cs // self.scale_factor
        else:
            lr_cs = (cs[0] // self.scale_factor, cs[1] // self.scale_factor)
#             print(lr_cs)
        lr_image = T.CenterCrop(lr_cs)(lr_image)
        lr_image = self.normalize(lr_image)
        return hr_image, lr_image
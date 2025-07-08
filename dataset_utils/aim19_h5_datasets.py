import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class h5dataset(Dataset):
    """"""
    def __init__(self, data_root='.../data/aim19/', partition='train', mode='hr',
                 transform=None, **kwargs):
        """
        Parameters
        ----------
        data_root : string, optional
            Directory of images.
        partition : string, optional
            Either 'train' or 'test'.
        mode : string, optional
            Either 'hr' or 'lr'.
        """  
        self.partition = partition
        self.mode = mode
        if mode == 'hr':
            self.crop_size = 128
        elif mode == 'lr':
            self.crop_size = 32
            
        if self.partition == 'train':
            self.ids = 200000
        elif self.partition == 'test':
            self.ids = 50000
        
        self.dir = '%s%s_%s%d'%(data_root, self.partition, self.mode, self.crop_size)
        

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of image.
            
        Returns
        -------
        y : torch.tensor
            High-resolution (HR) image in [0,1] or low-resolution (LR) image in [-1,1] (channels first).
        """
        with h5py.File(f'{self.dir}.h5', 'r') as file:
            y = file['imgs'][idx]
            
        return torch.tensor(y)
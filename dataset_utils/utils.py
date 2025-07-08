import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class h5dataset(Dataset):
    """"""
    def __init__(self, file_name='path-to-file', ids=50000, **kwargs):
        """
        Parameters
        ----------
        file_name : string, optional
            Path to hdf5 dataset.
        ids : int
            Dataset length.
        """  
        self.file_name = file_name
        self.ids = ids
        

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
        with h5py.File(self.file_name, 'r') as file:
            y = file['imgs'][idx]
            
        return torch.tensor(y)
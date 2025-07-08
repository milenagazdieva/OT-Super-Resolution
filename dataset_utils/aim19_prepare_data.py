import h5py
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

from tqdm.notebook import tqdm
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def save_to_HDF5(test = False, in_dir='.../data/aim19/', 
                 out_dir='.../data/aim19/', scaling_factor=4, 
                 crop_size=128, size=10, partition='train_hr', verbose=True):
    """Save images to HDF5.
    
    Parameters
    ----------
    scaling_factor : int
        The factor between shapes of low-resolution and high-resolution depth maps.
    img_size: int
        Size of the images
    crop_size : int
        HR crop size.
    partition : str
        Either train_hr, train_lr, test_hr or test_lr.
    """
    n_channels = 3
    
    if partition == 'train_hr':
        data_name = 'train_clean/'
    elif partition == 'train_lr':
        print('hi')
        data_name = 'train_noisy/'
    elif partition == 'test_hr':
        data_name = 'val_hr/'
    elif partition == 'test_lr':
        data_name = 'val_lr/'
    
    if partition == 'train_hr' or partition == 'test_hr':
        transform = transforms.Compose([transforms.RandomCrop(crop_size),
                                       transforms.ToTensor()])
    elif partition == 'train_lr' or partition == 'test_lr':
        print('hi')
        crop_size = crop_size // scaling_factor
        transform = transforms.Compose([transforms.RandomCrop(crop_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    file_path = out_dir + partition + '%d.h5'%crop_size
    dataset = datasets.ImageFolder(in_dir + data_name, 
                                   transform=transform)
    len_dataset = len(dataset)
    
    with h5py.File(f'{file_path}', 'w') as h:
        D = h.create_dataset('imgs', shape=(size, n_channels, crop_size, crop_size))
        for i in tqdm(range(size)) if verbose else range(size):
            idx = np.random.randint(0, len_dataset)
            x, _ = dataset[idx]
            D[i, :, :, :] = x
            
if __name__=='__main__':
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--PARTITION', type=str, default='test_lr')
    parser.add_argument('--OUT_DIR', type=str, default='.../data/aim19/')
    parser.add_argument('--SIZE', type=int, default=200000,
                        help='Dataset size.')
    parser.add_argument('--CROP_SIZE', type=int, default=128)
    parser.add_argument('--SCALE_FACTOR', type=int, default=4)
    args = parser.parse_args()
    
    PARTITION = args.PARTITION
    OUT_DIR = args.OUT_DIR
    SIZE = args.SIZE
    CROP_SIZE = args.CROP_SIZE
    SCALE_FACTOR = args.SCALE_FACTOR
    
    save_to_HDF5(partition=PARTITION, out_dir=OUT_DIR, size=SIZE, crop_size=CROP_SIZE, scaling_factor=SCALE_FACTOR)
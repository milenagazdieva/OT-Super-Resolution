import torch
import numpy as np
import random

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class DatasetSampler(Sampler):
    def __init__(self, dataset, num_workers=40, device='cuda'):
        super(DatasetSampler, self).__init__(device=device)
        
        self.shape = dataset[0][0].shape
        self.dim = np.prod(self.shape)
        loader = torch.utils.data.DataLoader(dataset, batch_size=num_workers, num_workers=num_workers)
        
        with torch.no_grad():
            self.dataset = torch.cat(
                [X for X in loader]
            )
        
    def sample(self, batch_size=16):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch
    
class Paired_DatasetSampler(Sampler):
    def __init__(self, dataset, num_workers=40, device='cuda'):
        super(Paired_DatasetSampler, self).__init__(device=device)
        
        self.shape = dataset[0][0].shape
        self.dim = np.prod(self.shape)
        loader = torch.utils.data.DataLoader(dataset, batch_size=num_workers, num_workers=num_workers)
        
        with torch.no_grad():
            self.X_dataset = torch.cat(
                [X for (X, Y) in loader]
            )
            self.Y_dataset = torch.cat(
                [Y for (X, Y) in loader]
            )
        
    def sample(self, batch_size=16):
        ind = random.choices(range(len(self.X_dataset)), k=batch_size)
        with torch.no_grad():
            X_batch = self.X_dataset[ind].clone().to(self.device).float()
            Y_batch = self.Y_dataset[ind].clone().to(self.device).float()
        return X_batch, Y_batch
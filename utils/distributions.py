import torch
import numpy as np
import random
from tqdm import tqdm

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
                [X for (X, y) in loader]
            )
        
    def sample(self, batch_size=16):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch
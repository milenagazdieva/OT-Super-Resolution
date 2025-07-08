#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
0
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import numpy as np
import torch
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import datasets
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
    
from .inception import InceptionV3
from .utils import freeze

def get_hr_inception_stats(dataset, batch_size=5, verbose=False):
    """Inception stats of HR images.
    
    Parameters
    ----------
        clean_dir : str
            Path to clean images folder
        size : int
            Size of the dataset for computing stats
        crop_size : int
            Cropped (square) size of the clean images
        batch_size : int
        verbose : str
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=20, shuffle=False, drop_last=False)
    
    # calculate Inception stats
    size = len(dataset)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    pred_arr = np.empty((size, dims))
    start_idx = 0
    
    for batch in tqdm(dataloader) if verbose else dataloader:
        batch = batch.type(torch.FloatTensor).cuda()
        
        with torch.no_grad():
            pred = model(batch)[0]
        pred = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx += pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    model = model.cpu()
    del model, pred_arr, pred, batch
    torch.cuda.empty_cache()
 
    return mu, sigma

def get_generated_inception_stats(G, dataset, batch_size=5, verbose=False, denormalize_first=False):
    """Inception stats of generated images.
    
    Parameters
    ----------
        G : torch.module
            Generator model.
        dataset : Dataset
            Dataset of LR images as torch.tensors in [-1, 1].
        batch_size : int
        verbose : str
        denormalize_first : bool
            If Generator unputs are in [0, 1], must set to True, False either.
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=20, shuffle=False, drop_last=False)
    
    # calculate Inception stats
    size = len(dataset)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    pred_arr = np.empty((size, dims))
    start_idx = 0
    
    for batch in tqdm(dataloader) if verbose else dataloader:
        batch = batch.type(torch.FloatTensor).cuda()
        
        if denormalize_first:
            batch = (batch + 1) / 2
            G_batch = G(batch)
        else:
            G_batch = ((G(batch) + 1) / 2)

        with torch.no_grad():
            pred = model(G_batch)[0]
        pred = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx += pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    model = model.cpu()
    del model, pred_arr, pred, batch
    torch.cuda.empty_cache()
 
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
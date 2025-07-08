import os
import torch
import torch.nn as nn
from torch.functional import F

import numpy as np
from PIL import Image

def normalize_(x):
    """Normalize RGB image.
    
    Parameters
    ----------
    x : array_like
        Input RGB image.
        
    Returns
    -------
    x : array_like
        Normalized `x`.
    """
    return (x-x.min())/(x.max()-x.min())

def calculate_valid_crop_size(crop_size, upscale_factor):
    """Calculates size of largest crop, divisible by upscale factor."""
    if isinstance(crop_size, int):
        return crop_size - (crop_size % upscale_factor)
    else:
        crop_size_w, crop_size_h = crop_size
        valid_crop_size_w = crop_size_w - (crop_size_w % upscale_factor)
        valid_crop_size_h = crop_size_h - (crop_size_h % upscale_factor)
        return (valid_crop_size_w, valid_crop_size_h)
    
def modcrop(x, scaling_factor=4):
    """Crop margins of an array to make the shape be divisible by scaling factor.
        
    Parameters
    ----------
    x : array_like
    Input array of 3-dimensional shape (W, H, 3).
    scaling_factor : int
    Factor to make the shape be divisible by.

    Returns
    -------
    x : array_like
    Cropped input array `x`.
    """
    size = x.shape[:2] - np.mod(x.shape[:2], scaling_factor)
    x = x[:size[0], :size[1], :]
    return x

def downsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = 1/scale_factor, mode='bicubic') # downsample
    return y

def upsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = scale_factor, mode='bicubic') # upsample
    return y

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def forward_chop(G, img, scale_factor=4, shave=20, min_size=256):
    if len(img.shape) == 4:
        img = img[0]
    w, h = img.shape[1], img.shape[2]
    if w < min_size and h < min_size:
        G_img = G(img[None])
        return G_img[0]
    else:
        img_0 = img[:, 0:w//2+shave, 0:h//2+shave]
        img_1 = img[:, 0:w//2+shave, h//2-shave:h]
        img_2 = img[:, w//2-shave:w, 0:h//2+shave]
        img_3 = img[:, w//2-shave:w, h//2-shave:h]
        G_imgs = []
        for img in [img_0, img_1, img_2, img_3]:
            G_img = forward_chop(G, img).detach().cpu()
            G_imgs.append(G_img)
        out = torch.zeros(img.shape[0], w*scale_factor, h*scale_factor)
        
        out[:, 0:w//2*scale_factor, 0:h//2*scale_factor] = G_imgs[0][:, 0:w//2*scale_factor, 0:h//2*scale_factor]
        out[:, 0:w//2*scale_factor, h//2*scale_factor:h*scale_factor] = G_imgs[1][:, 0:w//2*scale_factor, shave*scale_factor:]
        out[:, w//2*scale_factor:w*scale_factor, 0:h//2*scale_factor] = G_imgs[2][:, shave*scale_factor:, 0:h//2*scale_factor]
        out[:, w//2*scale_factor:w*scale_factor, h//2*scale_factor:h*scale_factor] = G_imgs[3][:, shave*scale_factor:, shave*scale_factor:]
        return out
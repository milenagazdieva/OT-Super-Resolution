import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lpips import LPIPS

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        
    def forward(self, in_1, in_2):
        return F.mse_loss(in_1, in_2, reduction='none').flatten(start_dim=1).mean(dim=1)
    
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        
    def forward(self, in_1, in_2):
        return F.l1_loss(in_1, in_2, reduction='none').flatten(start_dim=1).mean(dim=1)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_names=['3', '8', '15', '22']):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features
        self.layer_names = layer_names
        self.eval()
        
    def __call__(self, in_1, in_2):
        assert in_1.shape == in_2.shape
        loss = 0.
        out_1 = in_1; out_2 = in_2
        for name, module in self.vgg_layers._modules.items():
            out_1 = module(out_1); out_2 = module(out_2);
            if name in self.layer_names:
                loss += F.mse_loss(out_1, out_2, reduction='none').flatten(start_dim=1).mean(dim=1)
        return loss

class InjectiveVGGPerceptualLoss(nn.Module):
    def __init__(self, layer_names=['3', '8', '15', '22'], w_vgg=0.02, w_l2=1., w_l1=1./3):
        super(InjectiveVGGPerceptualLoss, self).__init__()
        self.w_vgg, self.w_l2, self.w_l1 = w_vgg, w_l2, w_l1
        self.vgg_loss = VGGPerceptualLoss(layer_names=layer_names)
        self.l2_loss = L2Loss()
        self.l1_loss = L1Loss()
        self.eval()
        
    def __call__(self, in_1, in_2):
        assert in_1.shape == in_2.shape
        return self.w_vgg * self.vgg_loss(in_1, in_2) + self.w_l2 * self.l2_loss(in_1, in_2) + self.w_l1 * self.l1_loss(in_1, in_2)
    
class VGGLPIPSPerceptualLoss(nn.Module):
    def __init__(self, layer_names=['3', '8', '15', '22'], w_vgg=0.02, w_l2=1., w_l1=1./3, w_lpips=0.1):
        super(VGGLPIPSPerceptualLoss, self).__init__()
        self.w_vgg, self.w_l2, self.w_l1, self.w_lpips = w_vgg, w_l2, w_l1, w_lpips
        self.vgg_loss = VGGPerceptualLoss(layer_names=layer_names)
        self.l2_loss = L2Loss()
        self.l1_loss = L1Loss()
        self.lpips = LPIPS()
        self.eval()
        
    def __call__(self, in_1, in_2):
        assert in_1.shape == in_2.shape
        return self.w_vgg * self.vgg_loss(in_1, in_2) + self.w_l2 * self.l2_loss(in_1, in_2) + self.w_l1 * self.l1_loss(in_1, in_2) + self.w_lpips * self.lpips(in_1, in_2)
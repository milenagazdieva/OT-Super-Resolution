import os

import torch
from torch.optim import Adam, Adamax
from torch.functional import F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import random
import argparse
from copy import deepcopy
import wandb

from utils.utils import (downsample, upsample, unfreeze, freeze, weights_init_D)
from utils.plotters import plot_imgs, plot_train_imgs, fig2data, fig2img
from utils.aim19_fid_score import (get_generated_inception_stats, get_hr_inception_stats, 
                                   calculate_frechet_distance)
from utils.aim19_distributions import *
from utils.losses import InjectiveVGGPerceptualLoss, VGGLPIPSPerceptualLoss
from dataset_utils.aim19_datasets import AugDataset, TestDataset
from dataset_utils.aim19_h5_datasets import h5dataset

from models.ResNet_D import ResNet_D
from models.edsr_G import EDSR

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
def train(D, G, experiment_name, G0_update=True, cost_name='L2', G_iters=1, G_lr=1e-4, D_lr=1e-4, optimizer='Adam', num_workers=6, batch_size=64, crop_size=128, scale_factor=4, fid_interval=1, plot_interval=1, max_steps=1):
    """Train an OT model."""
    
    lr_crop_size = crop_size // scale_factor
    if cost_name == 'L2':
        cost = F.mse_loss
    elif cost_name == 'VGG':
        cost = InjectiveVGGPerceptualLoss(w_vgg=w_vgg, w_l2=w_l2, w_l1=w_l1).cuda()
        freeze(cost)
    elif cost_name == 'PERP':
        cost = VGGLPIPSPerceptualLoss(w_vgg=w_vgg, w_l2=w_l2, w_l1=w_l1, w_lpips=w_lpips).cuda()
        freeze(cost)
    
    HR_dataset = AugDataset(datadir='.../data/aim19/train_clean/', crop_size=crop_size, 
                              flips=True, rotations=True)
    LR_dataset = AugDataset(datadir='.../data/aim19/train_noisy/', crop_size=lr_crop_size, 
                              flips=True, rotations=True)
    Test_dataset = TestDataset(hr_dir='.../data/aim19/val_hr/', 
                               lr_dir='.../data/aim19/val_lr/',
                               scale_factor=scale_factor, crop_size=crop_size)
    Full_test_dataset = TestDataset(hr_dir='.../data/aim19/val_hr/', 
                                    lr_dir='.../data/aim19/val_lr/',
                                    scale_factor=scale_factor, crop_size=None)
    fid_lr_dataset = h5dataset(partition='train', mode='lr')
    
    HR_dataloader = DataLoader(HR_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    LR_dataloader = DataLoader(LR_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    Test_dataloader = DataLoader(Test_dataset, batch_size=4, num_workers=num_workers, shuffle=False)
    
    X_iterator = iter(cycle(HR_dataloader))
    Y_iterator = iter(cycle(LR_dataloader))
    Test_iterator = iter(cycle(Test_dataloader))
    
    Y_train_fixed = next(Y_iterator).cuda()
    X_train_fixed = next(X_iterator).cuda()
    X_test_fixed, Y_test_fixed = next(Test_iterator)
    X_test_fixed = X_test_fixed.cuda()
    Y_test_fixed = Y_test_fixed.cuda()
    
    if optimizer == 'Adam':
        D_opt = torch.optim.Adam(D.parameters(), lr=D_lr, weight_decay=1e-10)
        G_opt = torch.optim.Adam(G.parameters(), lr=G_lr, weight_decay=1e-10)
    elif optimizer == 'Adamax':
        D_opt = torch.optim.Adamax(D.parameters(), lr=D_lr, weight_decay=1e-10)
        G_opt = torch.optim.Adamax(G.parameters(), lr=G_lr, weight_decay=1e-10)

    D_loss_history = []
    G_loss_history = []

    last_plot_step, last_fid_step = -np.inf, 0
    best_fid = np.inf
    
    try:
        stats = np.load('.../data/aim19/inc_stats_trainhr_s200000_cs%d.npz'%crop_size)
        mu_data, sigma_data = stats['mu'], stats['sigma']
    except:
        print('Saved HR inception stats not found. \nCalculating...')
        fid_hr_dataset = h5dataset(partition='train', mode='hr')
        mu_data, sidma_data = get_hr_inception_stats(fid_hr_dataset, batch_size=50, verbose=False)
    
    print('===> Training...')
    step = 0
    while step < max_steps:
        if step == 0:
            G0 = upsample;
        elif G0_update == True:
            freeze(G)
            G0 = deepcopy(G); freeze(G0)
        torch.cuda.empty_cache()
        D_loss_history = []
        for i in range(D_iters):
            G_loss_history = []
            unfreeze(G); freeze(D)
            for G_iter in range(G_iters):
                Y = next(Y_iterator).cuda()
                with torch.no_grad():
                    up_Y = G0(Y)
                G_opt.zero_grad()
                G_Y = G(Y)
                G_loss = .5 * cost(G_Y, up_Y).mean() - D(G_Y).mean()
                G_loss.backward(); G_opt.step()
                G_loss_history.append(G_loss.item())
                del G_loss, G_Y, up_Y, Y; torch.cuda.empty_cache()
            wandb.log({f'G loss' : np.sum(G_loss_history)}, step=step)
            del G_loss_history

            freeze(G); unfreeze(D);
            
            X = next(X_iterator).cuda()
            Y = next(Y_iterator).cuda()
            
            with torch.no_grad():
                G_Y = G(Y)
            D_opt.zero_grad()
            D_loss = D(G_Y).mean() - D(X).mean()
            D_loss.backward(); D_opt.step();
            wandb.log({f'D loss' : D_loss.item()}, step=step)
            del D_loss, Y, X, G_Y; torch.cuda.empty_cache()
            
            step += 1 # increase step
            
            if step >= last_plot_step + plot_interval:
                last_plot_step = step
                
                Y_train_random = next(Y_iterator).cuda()
                X_test_random, Y_test_random = next(Test_iterator)
                X_test_random = X_test_random.cuda()
                Y_test_random = Y_test_random.cuda()
    
                fig, axes = plot_imgs(Y_test_fixed, X_test_fixed, upsample, G)
                wandb.log({'Test Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig) 
                
                fig, axes = plot_imgs(Y_test_random, X_test_random, upsample, G)
                wandb.log({'Test Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)
                
                fig, axes = plot_train_imgs(Y_train_fixed, upsample, G)
                wandb.log({'Train Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)
                
                fig, axes = plot_train_imgs(Y_train_random, upsample, G)
                wandb.log({'Train Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

                del Y_train_random, X_test_random, Y_test_random

            if step >= last_fid_step + fid_interval:
                last_fid_step = step
                print('===> Get generated inception stats...')
                
                m, s = get_generated_inception_stats(G, fid_lr_dataset, batch_size=50, verbose=False)
                FID_G = calculate_frechet_distance(m, s, mu_data, sigma_data)
                wandb.log({f'FID_G' : FID_G.item()}, step=step)
                del m, s;  torch.cuda.empty_cache()

                if FID_G < best_fid:
                    best_fid = FID_G
                    freeze(G); freeze(D)
                    best_D_state_dict = D.state_dict()
                    best_G_state_dict = G.state_dict()
                    torch.save(best_D_state_dict, os.path.join('.../runs/', experiment_name+'/', 'best_state_D.pt'))
                    torch.save(best_G_state_dict, os.path.join('.../runs/', experiment_name+'/', 'best_state_G.pt'))
                    save_checkpoint('.../runs/', experiment_name, G, D, G_opt, D_opt, step, model_name='best_model.pt')
                
        freeze(G); freeze(D)        
        save_checkpoint('.../runs/', experiment_name, G, D, G_opt, D_opt, step, model_name='last_model.pt')
        
def save_checkpoint(save_dir, exp_name, G, D, optG, optD, step, model_name='model.pt'):
    save_path =  save_dir + exp_name + '/' + model_name
    params = {} # best params
    params['G'] = G.state_dict()
    params['D'] = D.state_dict()
    opt_params = {}
    opt_params['optG'] = optG.state_dict()
    opt_params['optD'] = optD.state_dict()
    state = {"step": step ,"model": params, 'optimizer': opt_params}
    torch.save(state, save_path)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--G_arch', type=str, default='EDSR',
                        help='Generator architecture: EDSR.')
    parser.add_argument('--n_resblocks', type=int, default=64,
                        help='number of residual blocks in EDSR')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='number of feature maps in EDSR')
    parser.add_argument('--res_scale', type=float, default=10,
                        help='residual scaling in EDSR')
    parser.add_argument('--G_init', type=str, default='None',
                        help='G initialization: kaiming or None')
    parser.add_argument('--D_init', type=str, default='kaiming',
                        help='D initialization: kaiming or None')
    parser.add_argument('--G0_update', type=bool, default=True, 
                        help='Update G0 on each step or not')
    parser.add_argument('--cost', type=str, default='L2', 
                        help='Type of cost: L2, L1, LPIPS, VGG or PERP')
    parser.add_argument('--w_l2', type=float, default=1., 
                        help='Weight of L2 loss in PERP pr VGG')
    parser.add_argument('--w_l1', type=float, default=1., 
                        help='Weight of L1 loss in PERP or VGG (*3)')
    parser.add_argument('--w_vgg', type=float, default=2., 
                        help='Weight of VGG loss in PERP or VGG (*100)')
    parser.add_argument('--w_lpips', type=float, default=2., 
                        help='Weight of LPIPS loss in PERP (*10)')
    parser.add_argument('--bs', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--cs', type=int, default=128, 
                        help='Crop size')
    parser.add_argument('--scale_factor', type=int, default=4, 
                        help='Scale factor')
    parser.add_argument('--steps', type=int, default=250000,
                        help='Maximum steps')
    parser.add_argument('--D_lr', type=float, default=10, 
                        help='D learning rate * 10**5')
    parser.add_argument('--G_lr', type=float, default=10, 
                        help='G learning rate * 10 ** 5')
    parser.add_argument('--D_iters', type=int, default=25000, 
                        help='Number of D steps')
    parser.add_argument('--G_iters', type=int, default=15, 
                        help='Number of G steps per one D step')
    parser.add_argument('--opt', type=str, default='Adam', 
                        help='Adam or Adamax')
    parser.add_argument('--n_workers', type=int, default=6)
    parser.add_argument('--fid', type=int, default=2000,
                        help='FID interval')
    parser.add_argument('--plot', type=int, default=200, 
                        help='Plot interval')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed')
    args = parser.parse_args()
    
    G0_update = args.G0_update
    cost = args.cost
    w_l2 = args.w_l2
    w_l1 = args.w_l1 / 3
    w_vgg = args.w_vgg / 100
    w_lpips = args.w_lpips / 10
    batch_size = args.bs
    crop_size = args.cs
    scale_factor = args.scale_factor
    max_steps = args.steps
    D_lr = args.D_lr / 10**5
    G_lr = args.G_lr / 10**5
    res_scale = args.res_scale / 10
    D_iters = args.D_iters
    G_iters = args.G_iters
    G_init = args.G_init
    D_init = args.D_init
    G_arch = args.G_arch
    optimizer = args.opt
    num_workers = args.n_workers
    fid_interval = args.fid                                      
    plot_interval = args.plot
    seed = args.seed
    
    today = datetime.today()
    md = today.strftime('%m%d')
    hm = today.strftime('%H%M%S')
    if args.G_arch == 'EDSR':
        G_name = 'EDSRr%df%ds%.3f'%(args.n_resblocks, args.n_feats, res_scale)
    
    if args.cost == 'L2':
        cost_name = args.cost
    elif args.cost == 'VGG':
        cost_name = args.cost + 'l2%.3f'%w_l2 + 'l1%.3f'%w_l1 + 'vgg%.3f'%w_vgg
    elif args.cost == 'PERP':
        cost_name = args.cost + 'l2%.3f'%w_l2 + 'l1%.3f'%w_l1 + 'vgg%.3f'%w_vgg + 'lpips%.3f'%w_lpips
    experiment_name = ('%s/v%s_7555d77_aim19_%s_G0upd%s_cost%s_Ginit%s_Dinit%s_bs%d_cs%d_scale%d_steps%d_Dlr%.4f_Glr%.4f_Diters%dK_Giters%d_opt%s_nworkers%d_fid%d_plot%d_seed%d'%(md, hm, G_name, str(G0_update), cost_name, G_init, D_init, batch_size, crop_size, scale_factor, max_steps, D_lr, G_lr, D_iters // 1000, G_iters, optimizer, num_workers, fid_interval, plot_interval, seed))                                                  
    output_path = os.path.join('.../runs/', experiment_name+'/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config = dict(
        dataset='AIM19',
        G_name=G_name,
        cost=cost_name,
        crop_size=crop_size,
        batch_size=batch_size,
        max_steps=max_steps,
        D_lr=D_lr, G_lr=G_lr,
        D_iters=D_iters, G_iters=G_iters,
        optimizer=optimizer,
        seed=seed
    )
    
    wandb.init(name=experiment_name, project="noname", entity="noname", config=config)
    
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if G_arch == 'EDSR':
        G = EDSR(n_resblocks=args.n_resblocks, n_feats=args.n_feats, res_scale=res_scale, 
                 scale_factor=args.scale_factor).cuda()

            
    D = ResNet_D(size=crop_size).cuda()
    if D_init == 'kaiming':
        D.apply(weights_init_D)
    
    train(D, G, experiment_name, G0_update=G0_update, cost_name=cost_name, G_iters=G_iters, G_lr=G_lr, D_lr=D_lr, optimizer=optimizer, num_workers=num_workers, batch_size=batch_size, crop_size=crop_size, 
scale_factor=scale_factor, fid_interval=fid_interval, plot_interval=plot_interval, max_steps=max_steps)
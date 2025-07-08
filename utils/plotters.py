import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from .utils import upsample, freeze

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def plot_train_imgs(Y, upsample, G, mean=0.5, std=0.5):
    freeze(G)
    with torch.no_grad():
        G_Y = G(Y) # model output
        G0_Y = upsample(Y) # upsampled version
    
    G0_Y = G0_Y.permute(0, 2, 3, 1).detach().cpu()
    G_Y = G_Y.permute(0, 2, 3, 1).detach().cpu()
    
    with torch.no_grad():
        imgs = torch.cat([G0_Y, G_Y]).mul(std).add(mean).numpy().clip(0,1)

    fig, axes = plt.subplots(2, Y.shape[0], figsize=(4*Y.shape[0], 4*2), dpi=100, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
    plot_params = dict(rotation='horizontal', va="center", labelpad=180, fontsize=55)
    axes[0, 0].set_ylabel('Bicubic \n upsample', **plot_params)
    axes[1, 0].set_ylabel('OTS (ours)', **plot_params)
    
    fig.tight_layout(h_pad=0.01, w_pad=0.01)
    torch.cuda.empty_cache();
    return fig, axes

def plot_imgs(Y, X, upsample, G, mean=0.5, std=0.5):
    freeze(G)
    with torch.no_grad():
        G_Y = G(Y) # model output
        G0_Y = upsample(Y) # upsampled version
    
    G0_Y = G0_Y.permute(0, 2, 3, 1).detach().cpu()
    G_Y = G_Y.permute(0, 2, 3, 1).detach().cpu()
    X = X.permute(0, 2, 3, 1).detach().cpu()
    
    with torch.no_grad():
        imgs = torch.cat([G0_Y, G_Y, X]).mul(std).add(mean).numpy().clip(0,1)

    fig, axes = plt.subplots(3, X.shape[0], figsize=(4*X.shape[0], 4*3), dpi=100, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
    plot_params = dict(rotation='horizontal', va="center", labelpad=180, fontsize=55)
    axes[0, 0].set_ylabel('Bicubic \n upsample', **plot_params)
    axes[1, 0].set_ylabel('OTS (ours)', **plot_params)
    axes[2, 0].set_ylabel('GT', **plot_params)
    
    fig.tight_layout(h_pad=0.01, w_pad=0.01)
    torch.cuda.empty_cache();
    return fig, axes


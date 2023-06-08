import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import numpy as np
import torch

def plot_test_ssh(l_im,save_path,date,cmap="coolwarm",save=True):

    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [inp,ssh12,target] = line
        min_val = min(torch.min(inp),torch.min(ssh12),torch.min(target))
        max_val = max(torch.max(inp),torch.max(ssh12),torch.max(target))
        im1 = axes[n,0].imshow(torch.squeeze(inp).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im2 = axes[n,1].imshow(torch.squeeze(ssh12).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im3 = axes[n,2].imshow(torch.squeeze(target).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)

        fig.colorbar(im1,ax=axes[n,0], orientation='vertical')
        fig.colorbar(im2,ax=axes[n,1], orientation='vertical')
        fig.colorbar(im3,ax=axes[n,1], orientation='vertical')

    
    cols=['ssh4','pred ssh12','true ssh12']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'images.png')
    plt.show()



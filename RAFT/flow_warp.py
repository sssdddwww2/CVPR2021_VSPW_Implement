import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.todevice(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
   
    return output


def flow_warp(x, flow):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
        padding_mode (str): 'zeros' or 'border'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[-2:]
    n, _, h, w = x.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid)





if __name__=='__main__':
    img = '/home/miaojiaxu/jiaxu3/vsp_segment/RAFT-master/demo-frames/frame_0016.png'
    img = Image.open(img)
    img = img.resize((1024,440))
    img = np.array(img)
    img = img/255.
   

   
    flow = np.load('tosave.npy')
    
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).permute(0,3,1,2)
    img = img.float()
    
    print(img.size())
    flow = torch.from_numpy(flow)
    flow = flow.unsqueeze(0)
    print(flow.size())
    img2 = warp(img,flow)
    print(img2.size())
    img2 = img2.squeeze(0).permute(1,2,0).numpy()
    print(img2.shape)
    img2 = Image.fromarray((img2*255.).astype('uint8'))
    img2.save('1.png')


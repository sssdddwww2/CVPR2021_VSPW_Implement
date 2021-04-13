import os
from PIL import Image
from RAFT_core.raft import RAFT
from RAFT_core.utils.utils import InputPadder
from collections import OrderedDict
from utils import Evaluator
import numpy as np
import torch
import torch.nn as nn
import sys

def flowwarp(x, flo):
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
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,mode='nearest',align_corners=False)

    return output



num_class=124

DIR_='/your/path/to/VSPW_480p'

data_dir=DIR_+'/data'
result_dir='./prediction'
#list_=['1001_5z_ijQjUf_0','1002_QXQ_QoswLOs']

split='val.txt'
with open(os.path.join(DIR_,split),'r') as f:

    list_ = f.readlines()
    list_ = [v[:-1] for v in list_]

###
gpu=0
model_raft = RAFT()
to_load = torch.load('./RAFT/models/raft-things.pth-no-zip')
new_state_dict = OrderedDict()
for k, v in to_load.items():
    name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
model_raft.load_state_dict(new_state_dict)
model_raft = model_raft.cuda(gpu)
###
total_TC=0.
evaluator = Evaluator(num_class)
for video in list_[:100]:
    if video[0]=='.':
        continue
    imglist_ = sorted(os.listdir(os.path.join(data_dir,video,'origin')))
    for i,img in enumerate(imglist_[:-1]):
        if img[0]=='.':
            continue
        #print('processing video : {} image: {}'.format(video,img))
        next_img = imglist_[i+1]
        imgname = img
        next_imgname = next_img
        img = Image.open(os.path.join(data_dir,video,'origin',img))
        next_img =Image.open(os.path.join(data_dir,video,'origin',next_img))
        image1 = torch.from_numpy(np.array(img))
        image2 = torch.from_numpy(np.array(next_img))
        padder = InputPadder(image1.size()[:2])
        image1 = image1.unsqueeze(0).permute(0,3,1,2)
        image2 = image2.unsqueeze(0).permute(0,3,1,2)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        image1 = image1.cuda(gpu)
        image2 = image2.cuda(gpu)
        with torch.no_grad():
            model_raft.eval()
            _,flow = model_raft(image1,image2,iters=20, test_mode=True)
            flow = padder.unpad(flow)

        flow = flow.data.cpu()
        pred = Image.open(os.path.join(result_dir,video,imgname.split('.')[0]+'.png'))
        next_pred = Image.open(os.path.join(result_dir,video,next_imgname.split('.')[0]+'.png'))
        pred =torch.from_numpy(np.array(pred))
        next_pred = torch.from_numpy(np.array(next_pred))
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
    #    print(next_pred)
        
        warp_pred = flowwarp(next_pred,flow)
    #    print(warp_pred)
        warp_pred = warp_pred.int().squeeze(1).numpy()
        pred = pred.unsqueeze(0).numpy()
        evaluator.add_batch(pred, warp_pred)
#    v_mIoU = evaluator.Mean_Intersection_over_Union()
#    total_TC+=v_mIoU
#    print('processed video : {} score:{}'.format(video,v_mIoU))

#TC = total_TC/len(list_)
TC = evaluator.Mean_Intersection_over_Union()

print("TC score is {}".format(TC))
        
print(split)
print(result_dir)

        
        
        
    


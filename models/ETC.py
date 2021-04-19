import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
from RAFT_core.utils.utils import InputPadder
from RAFT_core.raft import RAFT
from collections import OrderedDict
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
    output = nn.functional.grid_sample(x, vgrid,align_corners=False)

    return output



def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
         


class ETC(nn.Module):
    def __init__(self, net_enc, net_dec, crit, args,deep_sup_scale=None):
        super(ETC, self).__init__()

        self.raft = RAFT()
        to_load = torch.load('./RAFT_core/raft-things.pth-no-zip')
        new_state_dict = OrderedDict()
        for k, v in to_load.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        self.raft.load_state_dict(new_state_dict)
        ####
        self.mean=torch.FloatTensor([0.485, 0.456, 0.406])
        self.std=torch.FloatTensor([0.229, 0.224, 0.225])
        ####
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.args= args
        assert (self.args.clip_num==2 and self.args.dilation_num==0)
        self.conv_last_ = nn.Sequential(
            nn.Conv2d(2048+4*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, args.num_class, kernel_size=1)
             )
        self.criterion_flow = nn.MSELoss()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):

                        yield p

    def get_10x_lr_params(self):
        modules = [self.decoder,self.conv_last_]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p
    def get_1x_lr_params_bias(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p

    def get_10x_lr_params_bias(self):
        modules = [self.decoder,self.conv_last_]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p
    def forward(self, feed_dict, *, segSize=None):
        if feed_dict is None:
            return torch.zeros((0,self.args.num_class, 480, 720)).cuda()
        # training
        if segSize is None:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_num = len(clip_imgs)
            assert(clip_num==1)
            n,_,h,w = label.size()
            c_pre_img = clip_imgs[0]
            mean = self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mean = mean.to(c_img.device)
            mean = mean.expand_as(c_img)
            std = self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            std = std.to(c_img.device)
            std = std.expand_as(c_img)
            c_img_f = ((c_img*std)+mean)*255.
            c_pre_img_f = (c_pre_img*std+mean)*255.
            with torch.no_grad():
                self.raft.eval()
                padder = InputPadder((h,w))
                c_img_f_ = padder.pad(c_img_f)
                c_pre_img_f_ = padder.pad(c_pre_img_f)
                   # c_img_f = F.interpolate(c_img_f,(480,480),mode='bilinear',align_corners=False)
                   # c_pre_img_f = F.interpolate(c_pre_img_f,(480,480),mode='bilinear',align_corners=False)
                _,flow = self.raft(c_img_f_,c_pre_img_f_,iters=20, test_mode=True)
                flow = padder.unpad(flow)
                
            #########
            input = torch.cat([c_img,c_pre_img],0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            pred_deepsup_s,_,clip_tmp2 = self.decoder(clip_tmp)
            c_img_f2,c_pre_img_f2 = torch.split(clip_tmp2,split_size_or_sections=int(clip_tmp2.size(0)/2),dim=0)
            pred_ = self.conv_last_(clip_tmp2)
            c_pred_,c_pre_pred_ = torch.split(pred_,split_size_or_sections=int(pred_.size(0)/2),dim=0)

        
            c_pred_1 = nn.functional.log_softmax(c_pred_, dim=1)
            _,_,h,w = label.size()
            label= label.squeeze(1)
            label = label.long()
            c_pred_1 = F.interpolate(c_pred_1,(h,w),mode='bilinear',align_corners=False)
            loss = self.crit(c_pred_1,label)
            if self.deep_sup_scale is not None:
                pred_deepsup_s = torch.split(pred_deepsup_s,split_size_or_sections=int(pred_deepsup_s.size(0)/2),dim=0)
                pred_deepsup = F.interpolate(pred_deepsup_s[0],(h,w),mode='bilinear',align_corners=False)
                loss_deepsup = self.crit(pred_deepsup, label)
                loss = loss + loss_deepsup * self.deep_sup_scale
            flow = F.interpolate(flow,(h,w),mode='nearest')
            c_pre_pred_ = F.interpolate(c_pre_pred_,(h,w),mode='bilinear', align_corners=False)
            c_pred_ = F.interpolate(c_pred_,(h,w),mode='bilinear', align_corners=False)
            warp_i1 = flowwarp(c_pre_img, flow)
            warp_o1 = flowwarp(c_pre_pred_, flow)

            noc_mask2 = torch.exp(-1 * torch.abs(torch.sum(c_img - warp_i1, dim=1))).unsqueeze(1)
            ST_loss = self.args.st_weight * self.criterion_flow( c_pred_* noc_mask2, warp_o1 * noc_mask2)
            loss = loss+ST_loss
            acc = self.pixel_acc(c_pred_1, label)
            return loss, acc
        else:
            c_img = feed_dict['img_data']
            c_tmp = self.encoder(c_img,return_feature_maps=True)
            pred_deepsup_s,_,c_tmp2 = self.decoder(c_tmp)
            c_pred_ = self.conv_last_(c_tmp2)
            c_pred_ = nn.functional.interpolate(
                     c_pred_, size=segSize, mode='bilinear', align_corners=False)
            c_pred_ = nn.functional.softmax(c_pred_, dim=1)
            return c_pred_

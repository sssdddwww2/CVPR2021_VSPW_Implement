import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
from RAFT_core.raft import RAFT
from RAFT_core.utils.utils import InputPadder
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
class FlowCNN(nn.Module):
    def __init__(self):
        super(FlowCNN,self).__init__()
        self.conv1 = conv3x3_bn_relu(11,16)
        self.conv2 = conv3x3_bn_relu(16,32)
        self.conv3 = conv3x3_bn_relu(32,2)
        self.conv4 = conv3x3_bn_relu(4,2)
    def forward(self,img1,img2,flow):
        x = torch.cat([flow,img1,img2,img2-img1],1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([flow,x],1)
        x = self.conv4(x)
        return x
         


class NetWarp(nn.Module):
    def __init__(self, net_enc, net_dec, crit, args,deep_sup_scale=None):
        super(NetWarp, self).__init__()

        self.raft = RAFT()
        to_load = torch.load('./RAFT/models/raft-things.pth-no-zip')
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
        #assert (self.args.clip_num==2 and self.args.dilation_num==0)
        assert self.args.clip_num==2


        self.flowcnn=FlowCNN()
        self.conv_last_ = nn.Sequential(
            nn.Conv2d(2048+4*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, args.num_class, kernel_size=1)
             )
        self.w0_0 = nn.Parameter(torch.FloatTensor(2048), requires_grad=True) 
        self.w0_0.data.fill_(1.0)
        self.w0_1 = nn.Parameter(torch.FloatTensor(2048), requires_grad=True) 
        self.w0_1.data.fill_(0.0)
        self.w1_0 = nn.Parameter(torch.FloatTensor(4096), requires_grad=True) 
        self.w1_0.data.fill_(1.0)
        self.w1_1 = nn.Parameter(torch.FloatTensor(4096), requires_grad=True) 
        self.w1_1.data.fill_(0.0)

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
        modules = [self.decoder,self.flowcnn,self.conv_last_]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p
        for w in [self.w0_0,self.w1_0,self.w1_1,self.w0_1]:
            yield w

    def get_1x_lr_params_bias(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p

    def get_10x_lr_params_bias(self):
        modules = [self.decoder,self.flowcnn,self.conv_last_]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p
    def forward(self, feed_dict, *, segSize=None):
        if feed_dict is None:
            return torch.zeros((0,self.args.num_class, 480, 720)).cuda()
        # training
        c_img = feed_dict['img_data']
        clip_imgs = feed_dict['clipimgs_data']
        label = feed_dict['seg_label']
        clip_num = len(clip_imgs)
        assert(clip_num==1)
        n,_,h,w = label.size()
        c_pre_img = clip_imgs[0]
        mean= self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mean = mean.to(c_img.device)
        mean = mean.expand_as(c_img)
        std = self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = std.to(c_img.device)
        std = std.expand_as(c_img)
        c_img_f = ((c_img*std)+mean)*255.
        c_pre_img_f = (c_pre_img*std+mean)*255.
        with torch.no_grad():
            self.raft.eval()
#            if segSize is None:
            padder = InputPadder((h,w))
            c_img_f_ = padder.pad(c_img_f)
            c_pre_img_f_ = padder.pad(c_pre_img_f)
            _,flow = self.raft(c_img_f_,c_pre_img_f_,iters=20, test_mode=True)
            flow = padder.unpad(flow)
            

        #########
        #print(c_img_f.size())
        #c_img_show = c_img_f.squeeze(0).permute(1,2,0).cpu().numpy()
        #c_img_show = Image.fromarray(c_img_show.astype('uint8'))
        #c_img_show.save('111.png')
        #c_pre_img_show = c_pre_img_f.squeeze(0).permute(1,2,0).cpu().numpy()
        #c_pre_img_show = Image.fromarray(c_pre_img_show.astype('uint8'))
        #c_pre_img_show.save('222.png')
        #c_img_warp =flowwarp(c_pre_img_f,flow) 
        #c_img_warp = c_img_warp.squeeze(0).permute(1,2,0).cpu().numpy()
        #c_img_warp  = Image.fromarray(c_img_warp.astype('uint8'))
        #c_img_warp.save('warp_111.png')
        #exit()


        #########
        flow = self.flowcnn(c_img_f,c_pre_img_f,flow)
        input = torch.cat([c_img,c_pre_img],0)
        clip_tmp = self.encoder(input,return_feature_maps=True)
        c_img_f1,c_pre_img_f1 = torch.split(clip_tmp[-1],split_size_or_sections=int(clip_tmp[-1].size(0)/2),dim=0)
        flow_1 = F.interpolate(flow,c_img_f1.size()[-2:],mode='nearest')
        c_img_f1_warp = flowwarp(c_pre_img_f1,flow_1)
        new_c_img_f1 = self.w0_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(c_img_f1).to(c_img_f1.device)*c_img_f1+self.w0_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(c_img_f1_warp).to(c_img_f1_warp.device)*c_img_f1_warp
        feat = torch.cat([new_c_img_f1,c_pre_img_f1],0)
        clip_tmp[-1]=feat
        pred_deepsup_s,_,clip_tmp2 = self.decoder(clip_tmp)
        c_img_f2,c_pre_img_f2 = torch.split(clip_tmp2,split_size_or_sections=int(clip_tmp2.size(0)/2),dim=0)
        ####
        #ccc1,ccc2  = torch.split(_,split_size_or_sections=int(clip_tmp2.size(0)/2),dim=0)
        #save1 = ccc1.cpu().numpy()
        #save1 = np.save('1.npy',save1)
        #save2 = ccc2.cpu().numpy()
        #save2 = np.save('2.npy',save2)
        #exit()
        ###
        flow_2 = F.interpolate(flow,c_img_f2.size()[-2:],mode='nearest')
        c_img_f2_warp = flowwarp(c_pre_img_f2,flow_2)
        new_feat = self.w1_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(c_img_f2).to(c_img_f2)*c_img_f2+ \
                   self.w1_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(c_img_f2_warp).to(c_img_f2_warp)*c_img_f2_warp
        pred_ = self.conv_last_(new_feat)

        if segSize is not None:
            pred_ = nn.functional.interpolate(
                     pred_, size=segSize, mode='bilinear', align_corners=False)
            pred_ = nn.functional.softmax(pred_, dim=1)
            return pred_
        else:
            pred_ = nn.functional.log_softmax(pred_, dim=1)
            _,_,h,w = label.size()
            label= label.squeeze(1)
            label = label.long()
            pred_ = F.interpolate(pred_,(h,w),mode='bilinear',align_corners=False)
            loss = self.crit(pred_,label)
            if self.deep_sup_scale is not None:
                pred_deepsup_s = torch.split(pred_deepsup_s,split_size_or_sections=int(pred_deepsup_s.size(0)/2),dim=0)
                pred_deepsup = F.interpolate(pred_deepsup_s[0],(h,w),mode='bilinear',align_corners=False)
                #pred_deepsup= nn.functional.log_softmax(pred_deepsup, dim=1)
                loss_deepsup = self.crit(pred_deepsup, label)
                loss = loss + loss_deepsup * self.deep_sup_scale
            acc = self.pixel_acc(pred_, label)
            return loss, acc

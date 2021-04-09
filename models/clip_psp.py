import os
import torch
import torch.nn as nn
import torchvision
from models.sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class PPM_conv(nn.Module):
    def __init__(self,fc_dim=2048,num_class=None,pool_scales=(1,2,3,6)):
        super(PPM_conv,self).__init__()
        self.ppm = []
        for i in range(len(pool_scales)):
            self.ppm.append(nn.Sequential(
                                          nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                          BatchNorm2d(512),
                                          nn.ReLU(inplace=True)
                                         ))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last_ = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
            )



    def forward(self,x,xs):
        input_size = x.size()
        ppm_out = [x]
        for pool_scale,x_ in zip(self.ppm,xs):
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x_),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last_(ppm_out)
        return x

    
         
    


class Clip_PSP(nn.Module):
    def __init__(self, net_enc,  crit, args,pool_scales=(1, 2, 3, 6),deep_sup_scale=None):
        super(Clip_PSP, self).__init__()
        self.encoder = net_enc
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.args= args
        fc_dim=2048
        self.pool_scales = pool_scales
        self.ppm_conv = PPM_conv(fc_dim,args.num_class,pool_scales=pool_scales)
        self.deepsup  = nn.Sequential(
            nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3,
                      stride=1, padding=1, bias=False),
            BatchNorm2d(fc_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fc_dim // 4, args.num_class, 1, 1, 0),
            )
        self.ppm_pool = []
        if self.args.psp_weight:
            self.pspweight_conv = nn.Sequential(nn.Conv2d(fc_dim,1,kernel_size=1,bias=False),
                                                 nn.AdaptiveAvgPool2d((1,1)))
        for scale in pool_scales:
            self.ppm_pool.append(nn.AdaptiveAvgPool2d(scale))
        self.ppm_pool = nn.ModuleList(self.ppm_pool)
#        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
#        self.conv_last_deepsup_ = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
#        self.dropout_deepsup = nn.Dropout2d(0.1)

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
        modules = [self.ppm_conv]
        if self.deep_sup_scale is not None:
            modules.append(self.deepsup)
        if self.args.psp_weight:
            modules.append(self.pspweight_conv)
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
        modules = [self.ppm_conv]
        if self.deep_sup_scale is not None:
            modules.append(self.deepsup)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p
    def forward(self, feed_dict,  segSize=None):
        c_img = feed_dict['img_data']
        clip_imgs = feed_dict['clipimgs_data']
        label = feed_dict['seg_label']
        clip_num = len(clip_imgs)
        n,_,h,w = label.size()
        clip_imgs.append(c_img)
        input = torch.cat(clip_imgs,dim=0)
        clip_tmp = self.encoder(input,return_feature_maps=True)
        out_tmp = clip_tmp[-1]

        if self.args.psp_weight:
            psp_w = self.pspweight_conv(out_tmp)
            psp_w = torch.split(psp_w,split_size_or_sections=int(psp_w.size(0)/(clip_num+1)), dim=0)
            psp_w = [psp_ww.unsqueeze(-1) for psp_ww in psp_w]
            psp_w = torch.cat(psp_w,dim=-1)
            psp_w = F.softmax(psp_w,dim=-1)

        out_tmp = torch.split(out_tmp,split_size_or_sections=int(out_tmp.size(0)/(clip_num+1)), dim=0)
        c_tmp = out_tmp[-1]
        others_tmp = out_tmp[:-1]
        pooled_features=[]
        for i in range(len(self.pool_scales)):
            pooled_features.append([])
        for i,pool in enumerate(self.ppm_pool):
            tmp_f = pool(c_tmp)
            pooled_features[i].append(tmp_f.unsqueeze(-1))
        for i,pool in enumerate(self.ppm_pool):
            for j,other in enumerate(others_tmp):
                tmp_f = pool(other)
                pooled_features[i].append(tmp_f.unsqueeze(-1))

        #        if j==0:
        #            tmp_f = pool(other)
        #            pooled_features[i].append(tmp_f.unsqueeze(-1))
        #        elif j==1:
        #            if i>1:
        #                continue
        #            tmp_f = pool(other)
        #            pooled_features[i].append(tmp_f.unsqueeze(-1))
        #        else:
        #            if i!=0:
        #                continue
        #            tmp_f = pool(other)
        #            pooled_features[i].append(tmp_f.unsqueeze(-1))
        p_fs=[]
        for feature in pooled_features:
            feature  =torch.cat(feature,dim=-1)
            if self.args.psp_weight:
#                psp_w = psp_w.expand_as(feature)
                feature = feature * psp_w
            feature = torch.mean(feature,dim=-1)
            p_fs.append(feature)
        pred_ = self.ppm_conv(c_tmp,p_fs)        
        if segSize is not None:
            pred_ = nn.functional.interpolate(
                     pred_, size=segSize, mode='bilinear', align_corners=False)
            pred_ = nn.functional.softmax(pred_, dim=1)
            return pred_
        else:
            clip_labels = feed_dict['cliplabels_data']
            clip_labels.append(label)
            pred_ = nn.functional.log_softmax(pred_, dim=1)
            _,_,h,w = label.size()
            label= label.squeeze(1)
            label = label.long()
            pred_ = F.interpolate(pred_,(h,w),mode='bilinear',align_corners=False)
            loss = self.crit(pred_,label)
            alllabel = torch.cat(clip_labels,dim=0)
            if self.deep_sup_scale is not None:

                alllabel = alllabel.squeeze(1)
                alllabel = alllabel.long()
                conv_4 = clip_tmp[-2]
                
                pred_deepsup_s = self.deepsup(conv_4)
                pred_deepsup_s = nn.functional.log_softmax(pred_deepsup_s, dim=1)  
                pred_deepsup = F.interpolate(pred_deepsup_s,(h,w),mode='bilinear',align_corners=False)
                loss_deepsup = self.crit(pred_deepsup, alllabel)
                loss = loss + loss_deepsup * self.deep_sup_scale
            acc = self.pixel_acc(pred_, label)
            return loss, acc
        
            
        
                    
                     
                
                
             
        
    

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from models.ocr_modules.spatial_ocr_block import SpatialTemporalGather_Module, SpatialOCR_Module
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

#from lib.models.backbones.backbone_selector import BackboneSelector
#from lib.models.tools.module_helper import ModuleHelper


class ClipOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, net_enc, crit, args,deep_sup_scale=None):

        super(ClipOCRNet, self).__init__()

        self.args= args

        if self.args.use_memory:
            self.memory=[]
        self.crit = crit
        self.deep_sup_scale=deep_sup_scale
        self.encoder = net_enc
        self.inplanes = 128
        self.num_classes=args.num_class
        in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.spatial_context_head = SpatialTemporalGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05
                                                  )

        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

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
        modules = [self.conv_3x3,self.spatial_context_head,self.spatial_ocr_head,self.head,self.dsn_head]
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
        modules = [self.conv_3x3,self.spatial_context_head,self.spatial_ocr_head,self.head,self.dsn_head]
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

        
        x_dsn = self.dsn_head(clip_tmp[-2])
        out_tmp = clip_tmp[-1]
#        out_tmp = torch.split(out_tmp,split_size_or_sections=int(out_tmp.size(0)/(clip_num+1)), dim=0)
#        out_tmp = out_tmp[-1]

        out_tmp = self.conv_3x3(out_tmp)
        if segSize is not None:
            if self.args.use_memory:
                is_clean_memory=feed_dict['is_clean_memory']
                if is_clean_memory:
                    self.memory=[]
                context = self.spatial_context_head(out_tmp, x_dsn,clip_num,self.memory,self.args.memory_num)
            else:
                context = self.spatial_context_head(out_tmp, x_dsn,clip_num)

        else:
            context = self.spatial_context_head(out_tmp, x_dsn,clip_num)
        xs = torch.split(out_tmp,split_size_or_sections=int(out_tmp.size(0)/(clip_num+1)), dim=0)        

        if self.args.clipocr_all:
            x = self.spatial_ocr_head(out_tmp, context)
            x = self.head(x)
            if segSize is not None:  # is True during inference
                x = torch.split(x,split_size_or_sections=int(x.size(0)/(clip_num+1)), dim=0)
                x = x[-1]
                x = F.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = F.softmax(x, dim=1)
                return x
            else:
                clip_labels = feed_dict['cliplabels_data']
                clip_labels.append(label)
                pred_ = nn.functional.log_softmax(x, dim=1)
                alllabel = torch.cat(clip_labels,dim=0)
                _,_,h,w = alllabel.size()
                alllabel = alllabel.squeeze(1)
                alllabel = alllabel.long()
                pred_ = F.interpolate(pred_,(h,w),mode='bilinear',align_corners=False)
                loss = self.crit(pred_,alllabel)

                x_dsn = F.log_softmax(x_dsn, dim=1)
                x_dsn = F.interpolate(x_dsn,(h,w),mode='bilinear',align_corners=False)
                loss_deepsup = self.crit(x_dsn, alllabel)
                loss = loss+loss_deepsup*self.deep_sup_scale
                acc = self.pixel_acc(pred_, alllabel)

                return  loss,acc
            

        else:

            x = xs[-1]
    
            
            x = self.spatial_ocr_head(x, context)
            x = self.head(x)
    
            if segSize is not None:  # is True during inference
                x = F.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = F.softmax(x, dim=1)
                return x
            else:
                clip_labels = feed_dict['cliplabels_data']
                clip_labels.append(label)
                pred_ = nn.functional.log_softmax(x, dim=1)
                _,_,h,w = label.size()
                label= label.squeeze(1)
                label = label.long()
                pred_ = F.interpolate(pred_,(h,w),mode='bilinear',align_corners=False)
                loss = self.crit(pred_,label)
    
                alllabel = torch.cat(clip_labels,dim=0)
                alllabel = alllabel.squeeze(1)
                alllabel = alllabel.long()
                x_dsn = F.log_softmax(x_dsn, dim=1)
                x_dsn = F.interpolate(x_dsn,(h,w),mode='bilinear',align_corners=False)            
                loss_deepsup = self.crit(x_dsn, alllabel)
                loss = loss+loss_deepsup*self.deep_sup_scale
                acc = self.pixel_acc(pred_, label)
    
                return  loss,acc
    


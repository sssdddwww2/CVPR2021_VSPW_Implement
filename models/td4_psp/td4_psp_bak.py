import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18,resnet50,resnet101
import random
#import logging
#import pdb
import os
#from ptsemseg.loss import OhemCELoss2D,SegmentationLosses
from .transformer import Encoding, Attention
from models.sync_batchnorm import SynchronizedBatchNorm2d
from .loss import OhemCELoss2D
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, aux=True):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if aux:
            return conv_out[-2],x
        else:
            return x




class td4_psp(nn.Module):
    """
    """
    def __init__(self,
                 args,
                 norm_layer=BatchNorm2d,
                 backbone='resnet101',
                 dilated=True,
                 aux=True,
                 multi_grid=True,
                 loss_fn=None
                 ):
        super(td4_psp, self).__init__()

        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass =args.num_class 
        self.aux = aux
        self.args=args

        # copying modules from pretrained models
        self.backbone = backbone

        if backbone == 'resnet101':
            ResNet_ = resnet101
            self.expansion = 4
        elif backbone =='resnet50':
            ResNet_ = resnet50
            self.expansion = 4
        else:
            raise(ValueError)
       

        self.pre = ResNet_(pretrained=False)
        self.pretrained1 = ResnetDilated(self.pre)
        
        # bilinear upsample options

        self.psp1 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs)
        self.enc1 = Encoding(512*self.expansion*2,64,512*self.expansion,norm_layer)

 
        self.atn1_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_3 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_4 = Attention(512*self.expansion,64,norm_layer)
        
        
        self.layer_norm1 = Layer_Norm([int(args.cropsize/8)+1, int(args.cropsize/8)+1])

        self.head1 = FCNHead(512*self.expansion*1, self.nclass, norm_layer, chn_down=4)

        if aux:
            self.auxlayer1 = FCNHead(256*self.expansion, self.nclass, norm_layer)
            
        self.Q_queue = []
        self.V_queue = []
        self.K_queue = []

    def buffer_contral(self,q,k,v):
        assert(len(self.Q_queue)==len(self.V_queue))
        assert(len(self.Q_queue)==len(self.K_queue))

        self.Q_queue.append(q)
        self.V_queue.append(v)
        self.K_queue.append(k)

        if len(self.Q_queue)>3:
            self.Q_queue.pop(0)
            self.V_queue.pop(0)
            self.K_queue.pop(0)

        #self.pretrained_init_2p()


#        self.pretrained_init()


        #self.get_params()

    def get_1x_lr_params(self):
        modules = [self.pretrained1]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad =False
                    else:
                        if p.requires_grad and (not ('bias' in key)):

                            yield p

    def get_10x_lr_params(self):
        modules = [self.psp1,self.enc1,self.atn1_2,self.atn1_3,self.atn1_4,self.layer_norm1,self.head1]
        if  self.aux:
            modules.append(self.auxlayer1)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p

    def get_1x_lr_params_bias(self):
        modules = [self.pretrained1]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad=False
                    else:
                        if p.requires_grad and 'bias' in key:
                            yield p

    def get_10x_lr_params_bias(self):
        modules = [self.psp1,self.enc1,self.atn1_2,self.atn1_3,self.atn1_4,self.layer_norm1,self.head1]
        if  self.aux:
            modules.append(self.auxlayer1)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p
    
    def forward_path1(self, clip_imgs,segSize=None):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        
       
        if segSize is None:
            f1_img,f2_img,f3_img,f4_img=clip_imgs
            
            _, _, h, w = f4_img.size()
    
            
            c3_1,c4_1 = self.pretrained1(f4_img)
            c3_2,c4_2 = self.pretrained1(f1_img)
            c3_3,c4_3 = self.pretrained1(f2_img)
            c3_4,c4_4 = self.pretrained1(f3_img)
    
            z1 = self.psp1(c4_1)
            z2 = self.psp1(c4_2)
            z3 = self.psp1(c4_3)
            z4 = self.psp1(c4_4)
    
            v1, q1 = self.enc1(z1, pre=False)
            k_2, v_2, _ = self.enc1(z2, pre=True, start=True)
            k_3, v_3, q_3 = self.enc1(z3, pre=True)
            k_4, v_4, q_4 = self.enc1(z4, pre=True)
    
            v_3_ = self.atn1_2(k_2, v_2, q_3, fea_size=None)
            v_4_ = self.atn1_3(k_3, v_3_+v_3, q_4, fea_size=None)
            atn_1 = self.atn1_4(k_4, v_4_+v_4, q1, fea_size=z4.size())
            #atn_1 = self.atn4(k_4, v_4, q1, fea_size=z4.size())
    
            out1 = self.head1(self.layer_norm1(atn_1 + v1))
            out1_sub = self.head1(self.layer_norm1(v1))
    
            outputs1 = F.interpolate(out1, (h, w), **self._up_kwargs)
            outputs1_sub = F.interpolate(out1_sub, (h, w), **self._up_kwargs)
        
            #############Knowledge-distillation###########
            auxout1 = self.auxlayer1(c3_1)        
            auxout1 = F.interpolate(auxout1, (h, w), **self._up_kwargs)
            return outputs1, outputs1_sub, auxout1
        else:
            z1=self.psp1(self.pretrained1(clip_imgs,aux=False))

            q_cur,v_cur = self.enc1(z1, pre=False)

            if len(self.Q_queue)<3:
                output = self.head1(self.layer_norm1(v_cur))
            else:
                v_2_ = self.atn1_2(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
                v_3_ = self.atn1_3(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
                v_4_ = self.atn1_4(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z1.size())

            #pdb.set_trace()

                output = self.head1(self.layer_norm1(v_4_ + v_cur))

            q_cur,k_cur,v_cur = self.enc1(z1, pre=True)
            self.buffer_contral(q_cur,k_cur,v_cur)
            return output
        
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, feed_dict,segSize=None):
        if segSize is None:
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['cliplabels_data'][-1]
            lbl=label.squeeze(1).long()
            
            clip_num = len(clip_imgs)
            _,_,h,w = label.size()
            assert(clip_num==4)

            outputs = self.forward_path1(clip_imgs,segSize)
            outputs_, outputs_sub, auxout = outputs
            loss = self.loss_fn(outputs_,lbl) +\
                    0.5*self.loss_fn(outputs_sub,lbl) +\
                    0.1*self.loss_fn(auxout,lbl)

            acc  = self.pixel_acc(outputs_,lbl)
            return loss,acc
        else:
            c_img = feed_dict['img_data']
#            clip_imgs = feed_dict['clipimgs_data']
#            clip_imgs.append(c_img)
            outputs = self.forward_path1(c_img,segSize)
            outputs  =nn.functional.interpolate(
                               outputs, size=segSize, mode='bilinear', align_corners=False)
            outputs = nn.functional.softmax(outputs, dim=1)
            #outputs = F.interpolate(outputs, (1024, 2048), **self._up_kwargs)
            return outputs
        
        

    def pretrained_init(self):
        model_state = torch.load(self.args.predir)
        self.pretrained1.load_state_dict(model_state, strict=False)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
#        self.pid = pid
#        self.path_num = path_num
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
        self.init_weight()

    def forward(self, x):
        n, c, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)

#        x = x[:, self.pid*c//self.path_num:(self.pid+1)*c//self.path_num]
#        feat1 = feat1[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
#        feat2 = feat2[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
#        feat3 = feat3[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
#        feat4 = feat4[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]

        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, chn_down=4):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // chn_down

        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Layer_Norm(nn.Module):
    def __init__(self, shape):
        super(Layer_Norm, self).__init__()
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        return self.ln(x)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.LayerNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

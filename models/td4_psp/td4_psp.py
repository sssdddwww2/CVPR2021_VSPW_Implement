import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18,resnet50,resnet101
import random
#from ptsemseg.utils import split_psp_dict
#from ptsemseg.models.td4_psp.pspnet_4p import pspnet_4p
#import logging
#import pdb
import os
#from ptsemseg.loss import OhemCELoss2D,SegmentationLosses
from models.sync_batchnorm import SynchronizedBatchNorm2d
from .loss import OhemCELoss2D
from .transformer import Encoding, Attention

BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
#logger = logging.getLogger("ptsemseg")
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
                 backbone='resnet18',
                 aux=True,
                 loss_fn=None,
                 path_num=4
                 ):
        super(td4_psp, self).__init__()

        self.loss_fn = loss_fn
        self.path_num = path_num
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.args = args
        self.aux = aux
        nclass=args.num_class
        

        # copying modules from pretrained models
        self.backbone = backbone
        assert(path_num == 4)

        if backbone == 'resnet18':
            ResNet_ = resnet18
            deep_base = False
            self.expansion = 1
        elif backbone == 'resnet34':
            ResNet_ = resnet34
            deep_base = False
            self.expansion = 1
        elif backbone == 'resnet50':
            ResNet_ = resnet50
            deep_base = True
            self.expansion = 4
        else:
            raise RuntimeError("Four branch model only support ResNet18 amd ResNet34")

        self.pretrained1 = ResnetDilated(ResNet_(pretrained=False))
        self.pretrained2 = ResnetDilated(ResNet_(pretrained=False))
        self.pretrained3 = ResnetDilated(ResNet_(pretrained=False))
        self.pretrained4 = ResnetDilated(ResNet_(pretrained=False))
        #self.pretrained1 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
        #                                   deep_base=deep_base, norm_layer=norm_layer)
        #self.pretrained2 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
        #                                   deep_base=deep_base, norm_layer=norm_layer)
        #self.pretrained3 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
        #                                   deep_base=deep_base, norm_layer=norm_layer)
        #self.pretrained4 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
        #                                   deep_base=deep_base, norm_layer=norm_layer)
        # bilinear upsample options

        self.psp1 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=0)
        self.psp2 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=1)
        self.psp3 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=0)
        self.psp4 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=1)

        self.enc1 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc2 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc3 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc4 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)

 
        self.atn1_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_3 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn2_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn2_3 = Attention(512*self.expansion,64,norm_layer)
        self.atn2_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn3_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn3_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn3_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn4_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn4_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn4_3 = Attention(512*self.expansion,64,norm_layer)
        
        self.layer_norm1 = Layer_Norm([int(args.cropsize/8)+1, int(args.cropsize/8)+1])
        self.layer_norm2 = Layer_Norm([int(args.cropsize/8)+1, int(args.cropsize/8)+1])
        self.layer_norm3 = Layer_Norm([int(args.cropsize/8)+1, int(args.cropsize/8)+1])
        self.layer_norm4 = Layer_Norm([int(args.cropsize/8)+1, int(args.cropsize/8)+1])

        self.head1 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head2 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head3 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head4 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)

        if aux:
            self.auxlayer1 = FCNHead(256*self.expansion, nclass, norm_layer)
            self.auxlayer2 = FCNHead(256*self.expansion, nclass, norm_layer)
            self.auxlayer3 = FCNHead(256*self.expansion, nclass, norm_layer)
            self.auxlayer4 = FCNHead(256*self.expansion, nclass, norm_layer)
            
        #self.pretrained_init_2p()
#        self.pretrained_init()
        self.KLD = nn.KLDivLoss()
#        self.get_params()
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
    def pretrained_init(self):
        model_state = torch.load(self.args.predir)
        self.pretrained1.load_state_dict(model_state, strict=False)
        self.pretrained2.load_state_dict(model_state, strict=False)
        self.pretrained3.load_state_dict(model_state, strict=False)
        self.pretrained4.load_state_dict(model_state, strict=False)
    def get_1x_lr_params(self):
        modules = [self.pretrained1,self.pretrained2,self.pretrained3,self.pretrained4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad =False
                    else:
                        if p.requires_grad and (not ('bias' in key)):

                            yield p

    def get_10x_lr_params(self):
        modules = [self.psp1,self.psp2,self.psp3,self.psp4,self.enc1,self.enc2,self.enc3,self.enc4,self.atn1_2,self.atn1_3,self.atn1_4,self.atn2_1,self.atn2_3,self.atn2_4,self.atn3_1,self.atn3_2,self.atn3_4,self.atn4_1,self.atn4_2,self.atn4_3,self.layer_norm1,self.layer_norm2,self.layer_norm3,self.layer_norm4,self.head1,self.head2,self.head3,self.head4]
        if  self.aux:
            modules.append(self.auxlayer1)
            modules.append(self.auxlayer2)
            modules.append(self.auxlayer3)
            modules.append(self.auxlayer4)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p

    def get_1x_lr_params_bias(self):
        modules = [self.pretrained1,self.pretrained2,self.pretrained3,self.pretrained4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad=False
                    else:
                        if p.requires_grad and 'bias' in key:
                            yield p
    def get_10x_lr_params_bias(self):
        modules = [self.psp1,self.psp2,self.psp3,self.psp4,self.enc1,self.enc2,self.enc3,self.enc4,self.atn1_2,self.atn1_3,self.atn1_4,self.atn2_1,self.atn2_3,self.atn2_4,self.atn3_1,self.atn3_2,self.atn3_4,self.atn4_1,self.atn4_2,self.atn4_3,self.layer_norm1,self.layer_norm2,self.layer_norm3,self.layer_norm4,self.head1,self.head2,self.head3,self.head4]
        if  self.aux:
            modules.append(self.auxlayer1)
            modules.append(self.auxlayer2)
            modules.append(self.auxlayer3)
            modules.append(self.auxlayer4)
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
#        f1_img = f_img[0]
#        f2_img = f_img[1]
#        f3_img = f_img[2]
#        f4_img = f_img[3]
#        
#        _, _, h, w = f4_img.size()

            c3_1,c4_1 = self.pretrained1(f4_img)
            c3_2,c4_2 = self.pretrained2(f1_img)
            c3_3,c4_3 = self.pretrained3(f2_img)
            c3_4,c4_4 = self.pretrained4(f3_img)
    
            z1 = self.psp1(c4_1)
            z2 = self.psp2(c4_2)
            z3 = self.psp3(c4_3)
            z4 = self.psp4(c4_4)
    
            v1, q1 = self.enc1(z1, pre=False)
            k_2, v_2, _ = self.enc2(z2, pre=True, start=True)
            k_3, v_3, q_3 = self.enc3(z3, pre=True)
            k_4, v_4, q_4 = self.enc4(z4, pre=True)
    
            v_3_ = self.atn1_2(k_2, v_2, q_3, fea_size=None)
            v_4_ = self.atn1_3(k_3, v_3_+v_3, q_4, fea_size=None)
            atn_1 = self.atn1_4(k_4, v_4_+v_4, q1, fea_size=z4.size())
            #atn_1 = self.atn4(k_4, v_4, q1, fea_size=z4.size())
    
            out1 = self.head1(self.layer_norm1(atn_1 + v1))
            out1_sub = self.head1(self.layer_norm1(v1))
    
            outputs1 = F.interpolate(out1, (h, w), **self._up_kwargs)
            outputs1_sub = F.interpolate(out1_sub, (h, w), **self._up_kwargs)
            
           # #############Knowledge-distillation###########
           # self.teacher.eval()
           # T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
           # T_logit_1234 = T_logit_1234.detach()
           # T_logit_1 = T_logit_1.detach()
           # T_logit_2 = T_logit_2.detach()
           # T_logit_3 = T_logit_3.detach()
           # T_logit_4 = T_logit_4.detach()

           # KD_loss1 = self.KLDive_loss(out1,T_logit_1234)+ 0.5*self.KLDive_loss(out1_sub,T_logit_1)
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
        
    def forward_path2(self,clip_imgs,segSize=None):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        if segSize is None:
            f1_img,f2_img,f3_img,f4_img=clip_imgs
        #f1_img = f_img[0]
        #f2_img = f_img[1]
        #f3_img = f_img[2]
        #f4_img = f_img[3]
        
            _, _, h, w = f4_img.size()
    
            c3_1,c4_1 = self.pretrained1(f3_img)
            c3_2,c4_2 = self.pretrained2(f4_img)
            c3_3,c4_3 = self.pretrained3(f1_img)
            c3_4,c4_4 = self.pretrained4(f2_img)
    
            z1 = self.psp1(c4_1)
            z2 = self.psp2(c4_2)
            z3 = self.psp3(c4_3)
            z4 = self.psp4(c4_4)
    
            k_1, v_1, q_1 = self.enc1(z1, pre=True)
            v2, q2 = self.enc2(z2, pre=False)
            k_3, v_3, _ = self.enc3(z3, pre=True, start=True)
            k_4, v_4, q_4 = self.enc4(z4, pre=True)
    
            v_4_ = self.atn2_3(k_3, v_3, q_4, fea_size=None)
            v_1_ = self.atn2_4(k_4, v_4_+v_4, q_1, fea_size=None)
            atn_2 = self.atn2_1(k_1, v_1_+v_1, q2, fea_size=z4.size())
            #atn_2 = self.atn1(k_1, v_1, q2, fea_size=z4.size())
    
            ############### SegHead ##############
            out2 = self.head2(self.layer_norm2(atn_2 + v2))
            out2_sub = self.head2(self.layer_norm2(v2))
    
            outputs2 = F.interpolate(out2, (h, w), **self._up_kwargs)
            outputs2_sub = F.interpolate(out2_sub, (h, w), **self._up_kwargs)

            ##############Knowledge-distillation###########
            #self.teacher.eval()
            #T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            #T_logit_1234 = T_logit_1234.detach()
            #T_logit_1 = T_logit_1.detach()
            #T_logit_2 = T_logit_2.detach()
            #T_logit_3 = T_logit_3.detach()
            #T_logit_4 = T_logit_4.detach()

            #KD_loss2 = self.KLDive_loss(out2,T_logit_1234)+ 0.5*self.KLDive_loss(out2_sub,T_logit_2)
            auxout2 = self.auxlayer2(c3_2)        
            auxout2 = F.interpolate(auxout2, (h, w), **self._up_kwargs)
            return outputs2, outputs2_sub, auxout2
        else:
            z2=self.psp2(self.pretrained2(clip_imgs,aux=False))
    
            q_cur,v_cur = self.enc2(z2, pre=False)
    
            if len(self.Q_queue)<3:
                output = self.head2(self.layer_norm2(v_cur))
            else:
                v_2_ = self.atn2_3(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
                v_3_ = self.atn2_4(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
                v_4_ = self.atn2_1(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z2.size())
    
                output = self.head2(self.layer_norm2(v_4_ + v_cur))
    
            q_cur,k_cur,v_cur = self.enc2(z2, pre=True)
            self.buffer_contral(q_cur,k_cur,v_cur)
            return output

    def forward_path3(self,clip_imgs,segSize=None):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        if segSize is None:
            f1_img,f2_img,f3_img,f4_img=clip_imgs
#        f1_img = f_img[0]
#        f2_img = f_img[1]
#        f3_img = f_img[2]
#        f4_img = f_img[3]
                
            _, _, h, w = f4_img.size()
    
            c3_1,c4_1 = self.pretrained1(f2_img)
            c3_2,c4_2 = self.pretrained2(f3_img)
            c3_3,c4_3 = self.pretrained3(f4_img)
            c3_4,c4_4 = self.pretrained4(f1_img)
    
            z1 = self.psp1(c4_1)
            z2 = self.psp2(c4_2)
            z3 = self.psp3(c4_3)
            z4 = self.psp4(c4_4)
    
            k_1, v_1, q_1 = self.enc1(z1, pre=True)
            k_2, v_2, q_2 = self.enc2(z2, pre=True)
            v3, q3 = self.enc3(z3, pre=False)
            k_4, v_4, _ = self.enc4(z4, pre=True, start=True)
    
            v_1_ = self.atn3_4(k_4, v_4, q_1, fea_size=None)
            v_2_ = self.atn3_1(k_1, v_1_+v_1, q_2, fea_size=None)
            atn_3 = self.atn3_2(k_2, v_2_+v_2, q3, fea_size=z4.size())
            #atn_3 = self.atn2(k_2, v_2, q3, fea_size=z4.size())
    
            ############### SegHead ##############
            out3 = self.head3(self.layer_norm3(atn_3 + v3))
            out3_sub = self.head3(self.layer_norm3(v3))
    
            outputs3 = F.interpolate(out3, (h, w), **self._up_kwargs)
            outputs3_sub = F.interpolate(out3_sub, (h, w), **self._up_kwargs)

            #############Knowledge-distillation###########
            #self.teacher.eval()
            #T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            #T_logit_1234 = T_logit_1234.detach()
            #T_logit_1 = T_logit_1.detach()
            #T_logit_2 = T_logit_2.detach()
            #T_logit_3 = T_logit_3.detach()
            #T_logit_4 = T_logit_4.detach()

            #KD_loss3 = self.KLDive_loss(out3,T_logit_1234)+ 0.5*self.KLDive_loss(out3_sub,T_logit_3)
            auxout3 = self.auxlayer3(c3_3)        
            auxout3 = F.interpolate(auxout3, (h, w), **self._up_kwargs)
            return outputs3, outputs3_sub, auxout3
        else:
            z3 = self.psp3(self.pretrained3(clip_imgs,aux=False))
    
            q_cur,v_cur = self.enc3(z3, pre=False)
    
            if len(self.Q_queue)<3:
                output = self.head3(self.layer_norm3(v_cur))
            else:
                v_2_ = self.atn3_4(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
                v_3_ = self.atn3_1(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
                v_4_ = self.atn3_2(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z3.size())
    
                output = self.head3(self.layer_norm3(v_4_ + v_cur))
    
            q_cur,k_cur,v_cur = self.enc3(z3, pre=True)
            self.buffer_contral(q_cur,k_cur,v_cur)
            return output

    def forward_path4(self,clip_imgs,segSize=None):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        if segSize is None:
            f1_img,f2_img,f3_img,f4_img=clip_imgs
#        f1_img = f_img[0]
#        f2_img = f_img[1]
#        f3_img = f_img[2]
#        f4_img = f_img[3]
        
            _, _, h, w = f4_img.size()
    
            c3_1,c4_1 = self.pretrained1(f1_img)
            c3_2,c4_2 = self.pretrained2(f2_img)
            c3_3,c4_3 = self.pretrained3(f3_img)
            c3_4,c4_4 = self.pretrained4(f4_img)
    
            z1 = self.psp1(c4_1)
            z2 = self.psp2(c4_2)
            z3 = self.psp3(c4_3)
            z4 = self.psp4(c4_4)
    
            k_1, v_1, _ = self.enc1(z1, pre=True, start=True)
            k_2, v_2, q_2 = self.enc2(z2, pre=True)
            k_3, v_3, q_3 = self.enc3(z3, pre=True)
            v4, q4 = self.enc4(z4, pre=False)
    
            v_2_ = self.atn4_1(k_1, v_1, q_2, fea_size=None)
            v_3_ = self.atn4_2(k_2, v_2_+v_2, q_3, fea_size=None)
            atn_4 = self.atn4_3(k_3, v_3_+v_3, q4, fea_size=z4.size())
            #atn_4 = self.atn3(k_3, v_3, q4, fea_size=z4.size())
    
            ############### SegHead ##############
            out4 = self.head4(self.layer_norm4(atn_4 + v4))
            out4_sub = self.head4(self.layer_norm4(v4))
    
            outputs4 = F.interpolate(out4, (h, w), **self._up_kwargs)
            outputs4_sub = F.interpolate(out4_sub, (h, w), **self._up_kwargs)

            ##############Knowledge-distillation###########
            #self.teacher.eval()
            #T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            #T_logit_1234 = T_logit_1234.detach()
            #T_logit_1 = T_logit_1.detach()
            #T_logit_2 = T_logit_2.detach()
            #T_logit_3 = T_logit_3.detach()
            #T_logit_4 = T_logit_4.detach()

            #KD_loss4 = self.KLDive_loss(out4,T_logit_1234)+ 0.5*self.KLDive_loss(out4_sub,T_logit_4)
            auxout4 = self.auxlayer4(c3_4)        
            auxout4 = F.interpolate(auxout4, (h, w), **self._up_kwargs)
            return outputs4, outputs4_sub, auxout4
        else:
            z4 = self.psp4(self.pretrained4(clip_imgs,aux=False))
    
            q_cur,v_cur = self.enc4(z4, pre=False)
    
            if len(self.Q_queue)<3:
                output = self.head4(self.layer_norm4(v_cur))
            else:
                v_2_ = self.atn4_1(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
                v_3_ = self.atn4_2(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
                v_4_ = self.atn4_3(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z4.size())
    
                output = self.head4(self.layer_norm4(v_4_ + v_cur))
    
            q_cur,k_cur,v_cur = self.enc4(z4, pre=True)
            self.buffer_contral(q_cur,k_cur,v_cur)
            return output

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    def forward(self, feed_dict,segSize=None, pos_id=None):
        if segSize is None:
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['cliplabels_data'][-1]
            lbl=label.squeeze(1).long()

            clip_num = len(clip_imgs)
            _,_,h,w = label.size()
            assert(clip_num==4)
            if pos_id == 0:
                outputs = self.forward_path1(clip_imgs,segSize)
            elif pos_id == 1:
                outputs = self.forward_path2(clip_imgs,segSize)
            elif pos_id == 2:
                outputs = self.forward_path3(clip_imgs,segSize)
            elif pos_id == 3:
                outputs = self.forward_path4(clip_imgs,segSize)
            else:
                raise RuntimeError("Only Four Paths.")

            outputs_, outputs_sub, auxout = outputs
            loss = self.loss_fn(outputs_,lbl) +\
                    0.5*self.loss_fn(outputs_sub,lbl) +\
                    0.1*self.loss_fn(auxout,lbl) 
            acc  = self.pixel_acc(outputs_,lbl)

            return loss,acc
        else:
            c_img = feed_dict['img_data']
            if pos_id == 0:
                outputs = self.forward_path1(c_img,segSize)
            elif pos_id == 1:
                outputs = self.forward_path2(c_img,segSize)
            elif pos_id == 2:
                outputs = self.forward_path3(c_img,segSize)
            elif pos_id == 3:
                outputs = self.forward_path4(c_img,segSize)
            else:
                raise RuntimeError("Only Four Paths.")
            #outputs = F.interpolate(outputs, (1024, 2048), **self._up_kwargs)
            return outputs
        
        

#    def pretrained_init(self):
#        if self.psp_path is not None:
#            if os.path.isfile(self.psp_path):
#                logger.info("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
#                print("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
#                model_state = torch.load(self.psp_path)
#                backbone_state, psp_state, head_state1, head_state2, _, _, auxlayer_state = split_psp_dict(model_state,self.path_num//2)
#                self.pretrained1.load_state_dict(backbone_state, strict=True)
#                self.pretrained2.load_state_dict(backbone_state, strict=True)
#                self.pretrained3.load_state_dict(backbone_state, strict=True)
#                self.pretrained4.load_state_dict(backbone_state, strict=True)
#                self.psp1.load_state_dict(psp_state, strict=True)
#                self.psp2.load_state_dict(psp_state, strict=True)
#                self.psp3.load_state_dict(psp_state, strict=True)
#                self.psp4.load_state_dict(psp_state, strict=True)
#                self.head1.load_state_dict(head_state1, strict=False)
#                self.head2.load_state_dict(head_state2, strict=False)
#                self.head3.load_state_dict(head_state1, strict=False)
#                self.head4.load_state_dict(head_state2, strict=False)
#                self.auxlayer1.load_state_dict(auxlayer_state, strict=True)
#                self.auxlayer2.load_state_dict(auxlayer_state, strict=True)
#                self.auxlayer3.load_state_dict(auxlayer_state, strict=True)
#                self.auxlayer4.load_state_dict(auxlayer_state, strict=True)
#            else:
#                logger.info("No pretrained found at '{}'".format(self.psp_path))


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs, path_num=None, pid=None):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pid = pid
        self.path_num = path_num
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

        x = x[:, self.pid*c//self.path_num:(self.pid+1)*c//self.path_num]
        feat1 = feat1[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat2 = feat2[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat3 = feat3[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat4 = feat4[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]

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

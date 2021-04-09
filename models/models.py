import torch
import torch.nn as nn
import torchvision
from .BiConvLSTM import BiConvLSTM
from .non_local import NLBlockND
from .non_local_models import Non_local2d
from . import resnet, resnext, mobilenet, hrnet,hrnet_clip
from models.sync_batchnorm import SynchronizedBatchNorm2d
from .deeplab import DeepLab
from .warp_our import WarpNet
from .ocrnet import SpatialOCRNet

BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict=None, segSize=None):
        if feed_dict is None:
            return torch.zeros((0,self.args.num_class, 480, 720)).cuda()
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            label = feed_dict['seg_label']
            _,_,h,w = label.size()
            label = label.squeeze(1)
            label = label.long()
            pred = F.interpolate(pred,(h,w),mode='bilinear',align_corners=False)

            loss = self.crit(pred, label)
            if self.deep_sup_scale is not None:
        #        print('*'*100)
        #        print(self.deep_sup_scale)
                pred_deepsup = F.interpolate(pred_deepsup,(h,w),mode='bilinear',align_corners=False)
                loss_deepsup = self.crit(pred_deepsup, label)
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, label)
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred

        


class ClipWarpNet(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, args,deep_sup_scale=None):
        super(ClipWarpNet, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.args= args
        self.emb_dim=128
        self.prop_clip=WarpNet(args,fc_dim=self.emb_dim)
        self.last_layer= nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.emb_dim, self.args.num_class, kernel_size=1)
             )

    def get_1x_lr_params(self):
        if self.args.fix:
            modules = [self.encoder,self.decoder]
        else:
            modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad =False
                    else:
                        if p.requires_grad and (not ('bias' in key)):
                      
                            yield p

    def get_10x_lr_params(self):
        if self.args.fix:
            modules = [self.prop_clip,self.last_layer]
        else:
            modules = [self.decoder,self.prop_clip,self.last_layer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p

    def get_1x_lr_params_bias(self):
        if self.args.fix:
            modules = [self.encoder,self.decoder]
        else:
            modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad=False
                    else:
                        if p.requires_grad and 'bias' in key:
                            yield p

    def get_10x_lr_params_bias(self):
        if self.args.fix:
            modules = [self.prop_clip,self.last_layer]
        else:
            modules = [self.decoder,self.prop_clip,self.last_layer]
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
            clip_labels = feed_dict['cliplabels_data']
            clip_num = len(clip_imgs)
            _,_,h,w = label.size()
            #print(clip_num)
            
            if self.args.fix:
                with torch.no_grad():
                    clip_imgs.append(c_img)
                    clip_labels.append(label)
                    
                    input = torch.cat(clip_imgs,0)
                    alllabel = torch.cat(clip_labels,0)
                    clip_tmp = self.encoder(input,return_feature_maps=True)
                    pred_deepsup_s,clip_embs,_ = self.decoder(clip_tmp)

                pred_c,embs_ = self.prop_clip(clip_embs,clip_num+1)
                _,_,h,w = label.size()
                label= label.squeeze(1)
                label = label.long()
                pred_c = F.interpolate(pred_c,(h,w),mode='bilinear',align_corners=False)
                loss = self.crit(pred_c,label)
                acc = self.pixel_acc(pred_c, label)

                if self.args.allsup:

                    _,_,h,w = alllabel.size()
                    alllabel = alllabel.squeeze(1)
                    alllabel = alllabel.long()
                    
                    pred_s = self.last_layer(embs_)
                    pred_s = nn.functional.log_softmax(pred_s, dim=1)
                    pred_s = F.interpolate(pred_s,(h,w),mode='bilinear',align_corners=False)

                    loss_a = self.crit(pred_s, alllabel)

                    loss = loss + loss_a*self.args.allsup_scale


            else:
                clip_imgs.append(c_img)
                clip_labels.append(label)
                
                input = torch.cat(clip_imgs,0)
                alllabel = torch.cat(clip_labels,0)
                clip_tmp = self.encoder(input,return_feature_maps=True)
                pred_deepsup_s,clip_embs,_ = self.decoder(clip_tmp)

                pred_c,embs_ = self.prop_clip(clip_embs,clip_num+1)
                _,_,h,w = label.size()
                label= label.squeeze(1)
                label = label.long()
                pred_c = F.interpolate(pred_c,(h,w),mode='bilinear',align_corners=False)
                loss = self.crit(pred_c,label)
        
                if self.args.allsup:
                
                    _,_,h,w = alllabel.size()
                    alllabel = alllabel.squeeze(1)
                    alllabel = alllabel.long()
                    pred_s = self.last_layer(embs_)
                    pred_s = nn.functional.log_softmax(pred_s, dim=1)
                    pred_s = F.interpolate(pred_s,(h,w),mode='bilinear',align_corners=False)

                    loss_a = self.crit(pred_s, alllabel)
    
                    if self.deep_sup_scale is not None:
                        pred_deepsup = F.interpolate(pred_deepsup_s,(h,w),mode='bilinear',align_corners=False)
                        loss_deepsup = self.crit(pred_deepsup, alllabel)
                        loss = loss + (loss_a + loss_deepsup * self.deep_sup_scale)*self.args.allsup_scale
                    else:
                        loss = loss + loss_a*self.args.allsup_scale

                
                acc = self.pixel_acc(pred_c, label)
            return loss, acc
                

       # inference
        else:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            clip_num = len(clip_imgs)
            clip_imgs.append(c_img)

            input = torch.cat(clip_imgs,0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            pred_deepsup_s,clip_embs,_ = self.decoder(clip_tmp)

            pred_c = self.prop_clip(clip_embs,clip_num+1,segSize=segSize)


            return pred_c

class Conv_LSTM_Model(nn.Module):
    def __init__(self,args,input_size):
        super(Conv_LSTM_Model,self).__init__()
        num_classes = args.num_class
        EMBED_DIM=256
        emb_dim = EMBED_DIM
        self.args = args
        self.embed = nn.Conv2d(720,EMBED_DIM,kernel_size=3, stride=1, padding=1, bias=False)
        self.convlstm = BiConvLSTM(input_size, emb_dim, emb_dim, (3,3), 1)
        self.last_layer = nn.Conv2d(EMBED_DIM, num_classes, kernel_size=1, stride=1)

    def forward(self,clip_imgs):
        clip_num  = len(clip_imgs)
        input = torch.cat(clip_imgs,dim=0)

        emb = self.embed(input)
        embs = torch.split(emb,split_size_or_sections=int(emb.size(0)/clip_num), dim=0)
        embs = [emb.unsqueeze(1) for emb in embs]
        embs = torch.cat(embs,1)

        lstm_embs = self.convlstm(embs)
        embs = lstm_embs
        embs = torch.split(embs,1,dim=1)
        embs = [emb.squeeze(1) for emb in embs]
        embs = torch.cat(embs,dim=0)
        outputs = self.last_layer(embs)
        outputs = torch.split(outputs,split_size_or_sections=int(outputs.size(0)/clip_num),dim=0)

        return outputs



class Non_local(nn.Module):
    def __init__(self,args):
        super(Non_local,self).__init__()

        self.emb = nn.Conv2d(720,128,1,1)
        self.nonlocalblock = NLBlockND(in_channels=128, mode='dot', dimension=3, bn_layer=True)
        self.last_layer = nn.Conv2d(128, args.num_class, kernel_size=1, stride=1)
    def forward(self, clip_imgs):
        clip_num  = len(clip_imgs)
        input = torch.cat(clip_imgs,dim=0)

        emb = self.emb(input)
        embs = torch.split(emb,split_size_or_sections=int(emb.size(0)/clip_num), dim=0) 


        x = [xx.unsqueeze(2) for xx in embs]
        x = torch.cat(x,2)
        x = self.nonlocalblock(x)
        x = torch.split(x,1,dim=2)
        x = [xx.squeeze(2) for xx in x]
        x = torch.cat(x,dim=0)
        x = self.last_layer(x)
#            emb = self.embed(emb)
        x = torch.split(x,split_size_or_sections=int(x.size(0)/clip_num), dim=0)
        return x




class SegmentationModule_allclip(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, args,deep_sup_scale=None,inputsize=None):
        super(SegmentationModule_allclip, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        if args.convlstm:
            self.prop_clip = Conv_LSTM_Model(args,inputsize)
     
        elif args.non_local:
            self.prop_clip = Non_local(args)

        self.args= args

    def get_1x_lr_params(self):
        modules = [self.encoder,self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad =False
                    else:
                        if p.requires_grad and (not ('bias' in key)):
                      
                            yield p

    def get_10x_lr_params(self):
        modules = [self.prop_clip]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and (not ('bias' in key)):
                        yield p

    def get_1x_lr_params_bias(self):
        modules = [self.encoder,self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if self.args.fix:
                        p.requires_grad=False
                    else:
                        if p.requires_grad and 'bias' in key:
                            yield p

    def get_10x_lr_params_bias(self):
        modules = [self.prop_clip]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p


    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_num = len(clip_imgs)
            _,_,h,w = label.size()
            #print(clip_num)

             #######
            with torch.no_grad():
                self.encoder.eval()
                input2 = torch.cat(clip_imgs,0)
                clip_embs = self.encoder(input2,return_feature_maps=True)
                clip_embs = clip_embs[0].detach()
                clip_embs = [clip_embs]
                if not self.args.fix:
                    self.encoder.train()

            #print(clip_preds.size())

            clip_embs = torch.split(clip_embs[0],split_size_or_sections=int(clip_embs[0].size(0)/clip_num), dim=0)
            #print(len(clip_embs))
            #exit()

            labels_2 = feed_dict['cliplabels_data']
            preds__  =self.prop_clip(clip_embs)             

            losses = []
            for pred__,label_2 in zip(preds__,labels_2):
                label_2 = label_2.squeeze(1).long()
     
                pred__ = nn.functional.log_softmax(pred__, dim=1)
                pred__ = F.interpolate(pred__,(h,w),mode='bilinear',align_corners=False)
                loss_ = self.crit(pred__,label_2)
                losses.append(loss_)
            losses = sum(losses)/(len(losses))
            
            #loss = losses+0.5*loss1+loss2*0.3
            if self.args.fix:
                loss = losses


            else:
                loss = losses+0.5*loss1
                
            acc = self.pixel_acc(pred__, label_2)

            


            return loss, acc
        # inference
        else:
            final_pred = []
            clip_imgs = feed_dict['clipimgs_data']
            clip_num = len(clip_imgs)



            #######
            input2 = torch.cat(clip_imgs,0)
            clip_embs = self.encoder(input2,return_feature_maps=True)
            clip_embs = torch.split(clip_embs[0],split_size_or_sections=int(clip_embs[0].size(0)/clip_num), dim=0)

            final_pred = []

            for  pred__ in  preds__:
                pred__ = nn.functional.interpolate(
                               pred__, size=segSize, mode='bilinear', align_corners=False)
                pred__ = nn.functional.softmax(pred__, dim=1)
                final_pred.append(pred__)

            return final_pred



class SegmentationModule_clip(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule_clip, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        if feed_dict is None:
            return torch.zeros((0,self.args.num_class, 480, 720)).cuda()
        # training
        if segSize is None:
            #if self.deep_sup_scale is not None: # use deep supervision technique
            #    (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            #else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], feed_dict['clipimgs_data'],return_feature_maps=True))
            label = feed_dict['seg_label']
            _,_,h,w = label.size()
            label = label.squeeze(1)
            label = label.long()
            pred = F.interpolate(pred,(h,w),mode='bilinear',align_corners=False)

            loss = self.crit(pred, label)
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, label)
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'],feed_dict['clipimgs_data'] ,return_feature_maps=True,is_train=False), segSize=segSize)
            return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights='',args=None):
#        pretrained = True if len(weights) == 0 else False
        pretrained = False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        elif arch =='hrnetv2_clip':
            net_encoder = hrnet_clip.__dict__['hrnetv2_clip'](pretrained=pretrained,args=args)
       
        elif arch =='hrnetv2_clip2':
            net_encoder = hrnet_clip2.__dict__['hrnetv2_clip2'](pretrained=pretrained,args=args)

            if len(weights) > 0:
                print('Loading weights for net_encoder')
                net_encoder.hrnetv2.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if arch!='hrnetv2_clip2':
            if len(weights) > 0:
                print('Loading weights for net_encoder')
                net_encoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax
#                pool_scales=(6,)
                )
        elif arch =='ppm_deepsup_clip':
            net_decoder = PPMDeepsup_clip(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch =='ppm_clip':
            net_decoder = PPM_clip(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
 
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch =='deeplab':
            net_decoder = DeepLab(
                                  num_class=num_class,
                                  fc_dim=fc_dim,
                                  use_softmax=use_softmax
         
                                  )
        elif arch =='nonlocal2d':
            net_decoder = Non_local2d(num_class = num_class)
        elif arch=='ocrnet_deepsup':
            #print(arch)
            net_decoder = SpatialOCRNet(num_class = num_class) 
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

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

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


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

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

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

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last_ = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup_ = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last_(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup_(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last_1 = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
      
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last_1(x)

        if segSize is not None: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last_ = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup_ = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last_(ppm_out)

        if segSize is not None:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup_(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)

class PPMDeepsup_clip(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup_clip, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last_ = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.conv_last_deepsup_ = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        emb = self.conv_last_(ppm_out)

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup_(_)

        _ = nn.functional.log_softmax(_, dim=1)

        return  _,emb,ppm_out

class PPM_clip(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM_clip, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last_ = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True))

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        emb = self.conv_last_(ppm_out)

        return emb
# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last_ = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last_(fusion_out)

        if segSize is not None:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x

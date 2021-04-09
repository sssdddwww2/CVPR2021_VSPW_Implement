import torch
import torch.nn as nn
from .non_local import NLBlockND
import torch.nn.functional as F




class Non_local3d(nn.Module):
    def __init__(self,args,net_enc,crit,downsample=False):
        super(Non_local3d,self).__init__()
        self.encoder=net_enc
        self.downsample=downsample
        self.crit=crit
        self.emb = nn.Conv2d(2048,256,1,1)
        self.nonlocalblock = NLBlockND(in_channels=256, mode='dot', dimension=3, bn_layer=True)
        self.last_layer = nn.Conv2d(512, args.num_class, kernel_size=1, stride=1)

    def forward(self, feed_dict,segSize=None):
        clip_imgs = feed_dict['clipimgs_data']
        labels = feed_dict['cliplabels_data']
        clip_num  = len(clip_imgs)
        input = torch.cat(clip_imgs,dim=0)
        x = self.encoder(input,return_feature_maps=True)
        x = x[-1]

#        print(x.size())
        emb = self.emb(x)

        if self.downsample:
            h_,w_ = emb.size()[-2:]
            emb_ = F.avg_pool2d(emb,(2,2))
        else:
            emb_=emb
        embs = torch.split(emb_,split_size_or_sections=int(emb.size(0)/clip_num), dim=0)
        x = [xx.unsqueeze(2) for xx in embs]
        x = torch.cat(x,2)
        x = self.nonlocalblock(x)
        x = torch.split(x,1,dim=2)
        x = [xx.squeeze(2) for xx in x]
        x = torch.cat(x,dim=0)
        if self.downsample:
            x = F.interpolate(x,(h_,w_),mode='bilinear',align_corners=False)
        x = torch.cat((emb,x),dim=1) 
        x = self.last_layer(x)
#        emb = self.embed(emb)
        x = torch.split(x,split_size_or_sections=int(x.size(0)/clip_num), dim=0)
        if segSize is None:
            acc = []
            losses=[]
            for pred__,label_2 in zip(x,labels):
                _,_,h,w  = label_2.size()
                label_2 = label_2.squeeze(1).long()

                pred__ = nn.functional.log_softmax(pred__, dim=1)
                pred__ = F.interpolate(pred__,(h,w),mode='bilinear',align_corners=False)
                loss_ = self.crit(pred__,label_2)
                acc.append(self.pixel_acc(pred__, label_2))
                losses.append(loss_)
            losses = sum(losses)/(len(losses))
            acc = sum(acc)/len(acc)
            return losses,acc
        else:
            final_pred = []

            for  pred__ in  x:
                pred__ = nn.functional.interpolate(
                               pred__, size=segSize, mode='bilinear', align_corners=False)
                pred__ = nn.functional.softmax(pred__, dim=1)
                final_pred.append(pred__)

            return final_pred

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
        modules = [self.emb,self.nonlocalblock,self.last_layer]
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
        modules = [self.emb,self.nonlocalblock,self.last_layer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p


class Non_local2d(nn.Module):
    def __init__(self,num_class=None,downsample=False):
        super(Non_local2d,self).__init__()
#        self.encoder=net_enc
        self.downsample=downsample
#        self.crit=crit
        self.emb = nn.Conv2d(2048,256,1,1)
        self.nonlocalblock = NLBlockND(in_channels=256, mode='dot', dimension=2, bn_layer=True)
        self.last_layer = nn.Conv2d(512, num_class, kernel_size=1, stride=1)
    def forward(self,input, segSize=None):
        x=input[-1]
        #print(x.size())
        #exit()
#        x = self.encoder(input,return_feature_maps=True)
#        x = x[-1]

        emb = self.emb(x)
        _,c,h_,w_=emb.size()
        if self.downsample:
            x = F.avg_pool2d(emb,(2,2))
            x  =self.nonlocalblock(x)
            x = F.interpolate(x,(h_,w_),mode='bilinear',align_corners=False)
        else:
            x = self.nonlocalblock(emb)
        x = torch.cat([emb,x],dim=1)
        pred = self.last_layer(x)
        if segSize is None:
            pred =F.log_softmax(pred, dim=1)
#            loss = self.crit(pred, label)
#            acc = self.pixel_acc(pred, label)
#            return loss,acc
            return pred
        else:
            pred  = F.interpolate(
                     pred, size=segSize, mode='bilinear', align_corners=False)
            pred = F.softmax(pred, dim=1)
            return pred

#    def pixel_acc(self, pred, label):
#        _, preds = torch.max(pred, dim=1)
#        valid = (label >= 0).long()
#        acc_sum = torch.sum(valid * (preds == label).long())
#        pixel_sum = torch.sum(valid)
#        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
#        return acc
#    def get_1x_lr_params(self):
#        modules = [self.encoder]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and (not ('bias' in key)):
#
#                        yield p
#
#    def get_10x_lr_params(self):
#        modules = [self.emb,self.nonlocalblock,self.last_layer]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and (not ('bias' in key)):
#                        yield p
#
#    def get_1x_lr_params_bias(self):
#        modules = [self.encoder]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and 'bias' in key:
#                        yield p
#
#    def get_10x_lr_params_bias(self):
#        modules = [self.emb,self.nonlocalblock,self.last_layer]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and 'bias' in key:
#                        yield p

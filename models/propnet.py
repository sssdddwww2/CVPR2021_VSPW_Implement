import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from models.sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

def local_pairwise_map(x,y,max_distances):
    '''
    Args: x: [N,channel,height,width]
          y: [N,channel,height,width]
          distance: kernal=2*distance+1
    Return:
           dist: [N,h,w,k*k]
    '''
    n,c,h,w = x.size()
    x2 = x.view(n,c,-1).permute(0,2,1)
    x2 =torch.matmul(x2.unsqueeze(2),x2.unsqueeze(-1))

    y2 = y.view(n,c,-1).permute(0,2,1)
    y2 = torch.matmul(y2.unsqueeze(2),y2.unsqueeze(-1)).view(n,1,h,w)

    dist_maps=[]
    unfold_ys=[]
    for max_distance in max_distances:
        #print(max_distance)
        padded_y = F.pad(y, (max_distance, max_distance, max_distance, max_distance))
        padded_y2 = F.pad(y2, (max_distance, max_distance, max_distance, max_distance), mode='constant', value=1e20)

        kernel = 2*max_distance+1
        offset_y = F.unfold(padded_y, kernel_size=(h, w)).view(n,c, h*w,-1).permute(0,2,1, 3)
        offset_y2 = F.unfold(padded_y2, kernel_size=(h, w)).view(n,h,w,-1)
        x = x.contiguous().view(n,c, h * w, -1).permute(0,2, 3, 1)
        x2 = x2.view(n,h, w, 1)

        dists = x2 + offset_y2 - 2. * torch.matmul(x, offset_y).view(n,h, w,kernel*kernel)
        dists = (torch.sigmoid(dists) - 0.5) * 2
        dist_maps.append(dists)
    #    unfold_ys.append(offset_y.view(n,h,w,c,kernel,kernel))
    return dist_maps

def prop_pred(prev_frame_embedding, query_embedding, prev_frame_labels,
       max_distances,num_class=None):
    
    ds = local_pairwise_map(query_embedding, prev_frame_embedding,
                                   max_distances=max_distances)
    d = ds[0]
    
    max_distance = max_distances[0]
    N,_,height,width=prev_frame_embedding.size()
    prev_frame_labels = F.interpolate(prev_frame_labels.float(),(height,width),mode='nearest')
    labels = prev_frame_labels.float()
    padded_labels = F.pad(labels,( max_distance,  max_distance, max_distance,  max_distance),value=-1)
    offset_labels = F.unfold(padded_labels, kernel_size=(height, width)).view(N,height, width, -1, 1)
    gt_ids = torch.arange(num_class).to(labels.device)
    #print(gt_ids)
    #print(torch.unique(offset_labels))

    offset_masks = torch.eq(
                offset_labels,
                gt_ids.float().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0))

    
    d_tiled = d.unsqueeze(-1).repeat((1,1,1,1,gt_ids.size(0)))
    pad = torch.ones_like(d_tiled)
    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists,_ = torch.min(d_masked, dim=3)
    dists = dists.view(N, height, width, gt_ids.size(0)).permute(0,3,1,2)

    return dists

class _split_separable_conv2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=7):
        super(_split_separable_conv2d,self).__init__()
        self.conv1=nn.Conv2d(in_dim,in_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2),groups=in_dim)
        self.relu1=nn.ReLU(True)
        self.bn1 = BatchNorm2d(in_dim)
        self.conv2=nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1)
        self.relu2=nn.ReLU(True)
        self.bn2 = BatchNorm2d(out_dim)
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SegBlock(nn.Module):
    def __init__(self,in_dim,emb_dim,args):
        super(SegBlock,self).__init__()
        self.conv1 = _split_separable_conv2d(in_dim,emb_dim)
        self.conv2 = _split_separable_conv2d(emb_dim,emb_dim)
        self.conv3 = _split_separable_conv2d(emb_dim,emb_dim)
        self.conv4 = _split_separable_conv2d(emb_dim,emb_dim)
        self.last_layer = nn.Conv2d(emb_dim,args.num_class,kernel_size=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.last_layer(x)
        return x
        
    
class PropNet(nn.Module):
    def __init__(self,net_enc, net_dec, crit, args,emb_dim=256,deep_sup_scale=None):
        super(PropNet,self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.args = args
        self.deep_sup_scale=deep_sup_scale
        
        self.segblock = SegBlock(emb_dim+args.num_class,emb_dim,args)

        self.emb = conv3x3_bn_relu(512,emb_dim)
        self.emb2 = conv3x3_bn_relu(512,emb_dim)
        self.last_layer=nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(emb_dim, self.args.num_class, kernel_size=1))

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
        modules = [self.decoder,self.segblock,self.emb,self.last_layer]
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
        modules = [self.decoder,self.segblock,self.emb,self.last_layer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p
    def forward(self,feed_dict, segSize=None):
        if segSize is None:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_labels = feed_dict['cliplabels_data']
            clip_num = len(clip_imgs)
            n,_,h,w = label.size()
            clip_imgs.append(c_img)
            clip_labels.append(label)
            input = torch.cat(clip_imgs,dim=0)
            alllabel = torch.cat(clip_labels,dim=0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            pred_deepsup_s,clip_embs,_ = self.decoder(clip_tmp)
            embs_ = self.emb(clip_embs)
            _,_,h,w = alllabel.size()
            alllabel = alllabel.squeeze(1)
            alllabel = alllabel.long()
            pred_s = self.last_layer(embs_)
            pred_s = nn.functional.log_softmax(pred_s, dim=1)
            pred_s = F.interpolate(pred_s,(h,w),mode='bilinear',align_corners=False)
            
            pred_labels = torch.argmax(pred_s,dim=1)
            pred_labels  = torch.split(pred_labels,split_size_or_sections=int(pred_labels.size(0)/(clip_num+1)), dim=0)
            loss_a = self.crit(pred_s, alllabel)
            
            if self.deep_sup_scale is not None:
                pred_deepsup = F.interpolate(pred_deepsup_s,(h,w),mode='bilinear',align_corners=False)
                loss_deepsup = self.crit(pred_deepsup, alllabel)
                loss_a = (loss_a + loss_deepsup * self.deep_sup_scale)*self.args.allsup_scale
            embs_2 = self.emb2(clip_embs)
            embs_2 = torch.split(embs_2,split_size_or_sections=int(embs_.size(0)/(clip_num+1)), dim=0)
            c_emb = embs_2[-1]
            others = embs_2[:-1]
            losses = []
            _,_,h,w = label.size()
            label = label.squeeze(1)
            label = label.long()
            losses=[]
            for other,other_l in zip(others,pred_labels[:-1]):
                other_l = other_l.unsqueeze(1)
                prop_l = prop_pred(other,c_emb,other_l,self.args.max_distances,self.args.num_class)
                c_embemb = torch.cat([c_emb,prop_l],dim=1)
                pred_c = self.segblock(c_embemb)
                pred_c = nn.functional.log_softmax(pred_c, dim=1)
                pred_c = F.interpolate(pred_c,(h,w),mode='bilinear',align_corners=False)
                loss_ = self.crit(pred_c,label)
                losses.append(loss_)
            final_loss = sum(losses)/len(losses)+ loss_a
            acc = self.pixel_acc(pred_c, label)
            return final_loss,acc
        else:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_num = len(clip_imgs)
            clip_imgs.append(c_img)
            input = torch.cat(clip_imgs,dim=0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            pred_deepsup_s,clip_embs,_ = self.decoder(clip_tmp)
            embs_ = self.emb(clip_embs)
            pred_s = self.last_layer(embs_)
            pred_s = torch.split(pred_s,split_size_or_sections=int(pred_s.size(0)/(clip_num+1)), dim=0)
            pred_c = pred_s[-1]
            other_preds = pred_s[:-1]
            
            embs_2 = self.emb2(clip_embs)
            embs_2 = torch.split(embs_2,split_size_or_sections=int(embs_.size(0)/(clip_num+1)), dim=0)
            c_emb = embs_2[-1]
            others = embs_2[:-1]
            c_cpreds=[]
            c_cpreds.append(pred_c.unsqueeze(0))
            for other,other_l in zip(others,other_preds):
                other_l = torch.argmax(other_l,dim=1,keepdim=True)
                prop_l = prop_pred(other,c_emb,other_l,self.args.max_distances,self.args.num_class)
                c_embemb = torch.cat([c_emb,prop_l],dim=1)
                pred_c = self.segblock(c_embemb)
                c_cpreds.append(pred_c.unsqueeze(0))
            c_cpreds  = torch.cat(c_cpreds,dim=0)
            c_cpreds = torch.mean(c_cpreds,dim=0)
            c_cpreds = nn.functional.interpolate(
                         c_cpreds, size=segSize, mode='bilinear', align_corners=False)
            c_cpreds= nn.functional.softmax(c_cpreds, dim=1)
            return c_cpreds
 
        
           
        
            
            
        


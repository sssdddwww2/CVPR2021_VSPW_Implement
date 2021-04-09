import os
import torch
import torch.nn as nn
import torchvision
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F
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
        padded_y = F.pad(y, (max_distance, max_distance, max_distance, max_distance))
        padded_y2 = F.pad(y2, (max_distance, max_distance, max_distance, max_distance), mode='constant', value=1e20)

        kernel = 2*max_distance+1
        offset_y = F.unfold(padded_y, kernel_size=(h, w)).view(n,c, h*w,-1).permute(0,2,1, 3)
        offset_y2 = F.unfold(padded_y2, kernel_size=(h, w)).view(n,h,w,-1)
        x = x.contiguous().view(n,c, h * w, -1).permute(0,2, 3, 1)
        x2 = x2.view(n,h, w, 1)

        dists = x2 + offset_y2 - 2. * torch.matmul(x, offset_y).view(n,h, w,kernel*kernel)
        dist_maps.append(dists.view(n,h,w,1,kernel,kernel))
    #    unfold_ys.append(offset_y.view(n,h,w,c,kernel,kernel))
    return dist_maps


class OurWarpMerge(nn.Module):
    def __init__(self, net_enc, net_dec, crit, args,deep_sup_scale=None):
        super(OurWarpMerge, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.args= args
        emb_dim=256
        self.prop_clip=WarpNetMerge(args,emb_dim)
        self.last_layer= nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(emb_dim, self.args.num_class, kernel_size=1)
             )

    def forward(self, feed_dict, segSize=None):
        if segSize is None:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_labels = feed_dict['cliplabels_data']
            clip_num = len(clip_imgs)
            clip_imgs.append(c_img)
            clip_labels.append(label)

            input = torch.cat(clip_imgs,0)
            alllabel = torch.cat(clip_labels,0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            clip_embs = self.decoder(clip_tmp)
            pred_cs,embs_,pred_deepsup_s= self.prop_clip(clip_embs,clip_tmp[-2],clip_num+1)

            _,_,h,w = alllabel.size()
            alllabel = alllabel.squeeze(1)
            alllabel = alllabel.long()
            pred_s = self.last_layer(embs_)
            pred_s = nn.functional.log_softmax(pred_s, dim=1)
            pred_s = F.interpolate(pred_s,(h,w),mode='bilinear',align_corners=False)

            loss_a = self.crit(pred_s, alllabel)

#            if self.deep_sup_scale is not None:
            pred_deepsup_s=nn.functional.log_softmax(pred_deepsup_s, dim=1)
            pred_deepsup = F.interpolate(pred_deepsup_s,(h,w),mode='bilinear',align_corners=False)
            loss_deepsup = self.crit(pred_deepsup, alllabel)
            loss_ = (loss_a + loss_deepsup) * self.deep_sup_scale
            losses=[]
            _,_,h,w = label.size()
            label = label.squeeze(1)
            label = label.long()
            for pred_c in pred_cs:
                pred_c =nn.functional.log_softmax(pred_c, dim=1)
                pred_c = F.interpolate(pred_c,(h,w),mode='bilinear',align_corners=False)
                losses.append(self.crit(pred_c, label))
            loss = sum(losses)/len(losses)+loss_
            acc = self.pixel_acc(pred_c, label)
            return loss, acc
        else:
            c_img = feed_dict['img_data']
            clip_imgs = feed_dict['clipimgs_data']
            label = feed_dict['seg_label']
            clip_num = len(clip_imgs)
            clip_imgs.append(c_img)

            input = torch.cat(clip_imgs,0)
            clip_tmp = self.encoder(input,return_feature_maps=True)
            clip_embs = self.decoder(clip_tmp)
            #pred_cs,embs_= self.prop_clip(clip_embs,clip_num+1)
            pred_cs,embs_,pred_deepsup_s= self.prop_clip(clip_embs,clip_tmp[-2],clip_num+1)

            pred_s = self.last_layer(embs_)
            pred_cc = torch.split(pred_s,split_size_or_sections=int(pred_s.size(0)/(clip_num+1)), dim=0)
            pred_cc = pred_cc[-1]

            final_preds=[pred_cc.unsqueeze(0)]
            for pred_c in pred_cs:
                final_preds.append(pred_c.unsqueeze(0))
            final_preds = torch.cat(final_preds,dim=0)
            final_preds = torch.mean(final_preds,dim=0)
            final_preds = nn.functional.interpolate(
                         final_preds, size=segSize, mode='bilinear', align_corners=False)
            final_preds= nn.functional.softmax(final_preds, dim=1)


            return final_preds
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
        modules = [self.decoder,self.prop_clip,self.last_layer]
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
        modules = [self.decoder,self.prop_clip,self.last_layer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for key, p in m[1].named_parameters():
                    if p.requires_grad and 'bias' in key:
                        yield p



class WarpNetMerge(nn.Module):
    def __init__(self,args,emb_dim=256):
        super(WarpNetMerge,self).__init__()
        self.args = args
        self.emb = conv3x3_bn_relu(512,emb_dim)
        self.emb2 = conv3x3_bn_relu(1024,emb_dim)
        self.emb_dim = emb_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last_layer=nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(emb_dim, self.args.num_class, kernel_size=1))
         
        self.last_layer2=nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(emb_dim*2, self.args.num_class, kernel_size=1))
        

    def forward(self,clip_embs,conv4,clip_num,segSize=None):
        #print('*'*100)
        clip_emb2 = self.emb(clip_embs)
        clip_emblist = torch.split(clip_emb2,split_size_or_sections=int(clip_emb2.size(0)/clip_num), dim=0)
        c_img = clip_emblist[-1]
        other_clips = clip_emblist[:-1]

        conv4_emb = self.emb2(conv4)

        conv4_emblist = torch.split(conv4_emb,split_size_or_sections=int(conv4_emb.size(0)/clip_num), dim=0)
        c_conv4 = conv4_emblist[-1]
        other_conv4 = conv4_emblist[:-1]

        all_dist_maps=[]
        for other in other_conv4:
            dist_maps = local_pairwise_map(c_conv4,other,self.args.max_distances)
            all_dist_maps.append(dist_maps)

        n,c,h,w = c_img.size()
        ####
        deepsup = self.last_layer(conv4_emb)

        ####
        final_preds = []
        for other,distmaps in zip(other_clips,all_dist_maps):
            warp_fs=[]
            for max_distance,dist_map in zip(self.args.max_distances,dist_maps):
                kernel=2*max_distance+1
                other_ = F.pad(other,(max_distance, max_distance, max_distance, max_distance))
                other_unfold = F.unfold(other_,kernel_size=(h, w)).view(n,c, h * w,-1).permute(0,2,1, 3).view(n,h,w,c,kernel,kernel)
                if self.args.distsoftmax:
                
                    dist_map = dist_map.view(n,h,w,1,kernel*kernel)
                    dist_map = F.softmax(1./(dist_map*self.args.temp+1e-5),4)
                    dist_map = dist_map.view(n,h,w,1,kernel,kernel)
                    warp_feature = other_unfold*dist_map 
              
                    warp_feature = warp_feature.contiguous().view(n,h*w*c,kernel,kernel)  
                    warp_feature = self.avgpool(warp_feature)
                    warp_feature = warp_feature.view(n,h,w,c,1).permute(0,3,1,2,4)

                elif self.args.distnearest:
                    other_unfold = other_unfold.view(n,h,w,c,kernel*kernel)
                    dist_map = dist_map.view(n,h,w,1,kernel*kernel)
                    _,dist_map_index = torch.max(dist_map,dim=4,keepdim=True)
                    dist_map_index = dist_map_index.repeat(1,1,1,self.emb_dim,1)
                    warp_feature = other_unfold.gather(-1,dist_map_index)
                    warp_feature = warp_feature.permute(0,3,1,2,4)
                
                else:
                    dist_map =1- (torch.sigmoid(dist_map) - 0.5) * 2
                    warp_feature = other_unfold*dist_map 
                    warp_feature = warp_feature.contiguous().view(n,h*w*c,kernel,kernel)  
                    warp_feature = self.avgpool(warp_feature)
                    warp_feature = warp_feature.view(n,h,w,c,1).permute(0,3,1,2,4)
                warp_fs.append(warp_feature)
            warp_fs = torch.cat(warp_fs,-1)
            warp_fs = torch.mean(warp_fs,dim=-1)
            final_fea=torch.cat((c_img,warp_fs),dim=1)
            x = self.last_layer2(final_fea)
            final_preds.append(x)
        
        
            return final_preds,clip_emb2,deepsup
        
       
       
                
        
        
if __name__=='__main__':
    a = torch.rand(2,100,50,60)
    b = torch.rand(2,100,50,60)
    dist = local_pairwise_map(a,b,3)
    print(dist.size())
    print(ofb.size())

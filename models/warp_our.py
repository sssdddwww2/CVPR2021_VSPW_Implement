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
#def local_pairwise_map(x,y,max_distances):
#    '''
#    Args: x: [N,channel,height,width]
#          y: [N,channel,height,width]
#          distance: kernal=2*distance+1
#    Return:
#           dist: [N,h,w,k,k]
#    '''
#    n,c,h,w = x.size()
#    x2 = x.view(n,c,-1).permute(0,2,1)
#    x2 =torch.matmul(x2.unsqueeze(2),x2.unsqueeze(-1))
#
#    y2 = y.view(n,c,-1).permute(0,2,1)
#    y2 = torch.matmul(y2.unsqueeze(2),y2.unsqueeze(-1)).view(n,1,h,w)
#
#    dist_maps=[]
#    unfold_ys=[]
#    for max_distance in max_distances:
#        padded_y = F.pad(y, (max_distance, max_distance, max_distance, max_distance))
#        padded_y2 = F.pad(y2, (max_distance, max_distance, max_distance, max_distance), mode='constant', value=1e20)
#    
#        kernel = 2*max_distance+1
#        offset_y = F.unfold(padded_y, kernel_size=(kernel, kernel)).view(n,c, kernel*kernel,h * w).permute(0,3,1, 2)
#        offset_y2 = F.unfold(padded_y2, kernel_size=(kernel, kernel)).view(n,kernel*kernel,h,w).permute(0,2,3,1)
#        x = x.contiguous().view(n,c, h * w, -1).permute(0,2, 3, 1)
#        x2 = x2.view(n,h, w, 1)
#    
#        dists = x2 + offset_y2 - 2. * torch.matmul(x, offset_y).view(n,h, w,kernel*kernel)
#        dist_maps.append(dists.view(n,h,w,1,kernel,kernel))
#        unfold_ys.append(offset_y.view(n,h,w,c,kernel,kernel))
#    return dist_maps,unfold_ys


class WarpNet(nn.Module):
    def __init__(self,args,fc_dim=128,emb_dim=256):
        super(WarpNet,self).__init__()
        self.args = args
        self.emb = conv3x3_bn_relu(512,emb_dim)
        self.emb_2 = conv3x3_bn_relu(512,fc_dim)
        self.emb_dim = emb_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last_layer=nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(emb_dim, self.args.num_class, kernel_size=1))
        if self.args.linear_combine:
            
            self.ws = []
            for i in range(self.args.clip_num):
                self.ws.append(torch.nn.Parameter(torch.FloatTensor(emb_dim), requires_grad=True))
                self.register_parameter('w{}'.format(i), self.ws[i])
                if i==0:
                    self.ws[i].data.fill_(1.)
                else:
                    self.ws[i].data.fill_(0.2)
         
        

    def forward(self,clip_embs,clip_num,segSize=None):
        clip_emb2 = self.emb_2(clip_embs)
        clip_emblist = torch.split(clip_emb2,split_size_or_sections=int(clip_emb2.size(0)/clip_num), dim=0)
        c_img = clip_emblist[-1]
        other_clips = clip_emblist[:-1]
        all_dist_maps=[]
        for other in other_clips:
            dist_maps = local_pairwise_map(c_img,other,self.args.max_distances)
            all_dist_maps.append(dist_maps)

        clip_emb_s = self.emb(clip_embs)
        clip_emb_s = torch.split(clip_emb_s,split_size_or_sections=int(clip_emb_s.size(0)/clip_num), dim=0)
        c_img = clip_emb_s[-1]
        n,c,h,w = c_img.size()
        other_clips=clip_emb_s[:-1]

        final_emb = [c_img]
        for other,distmaps in zip(other_clips,all_dist_maps):
            warp_fs=[]
            for max_distance,dist_map in zip(self.args.max_distances,dist_maps):
                kernel=2*max_distance+1
                other_ = F.pad(other,(max_distance, max_distance, max_distance, max_distance))
                other_unfold = F.unfold(other_,kernel_size=(h,w)).view(n,c, h * w,kernel*kernel).permute(0,2,1, 3).view(n,h,w,c,kernel,kernel)
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
            final_emb.append(warp_fs)
        
        
        final_fea=[]
        if  self.args.linear_combine:
             for i in range(self.args.clip_num):
                 ww = self.ws[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                  
                 ww = ww.expand_as(c_img)
                 ww = ww.to(c_img.device)
                 final_fea.append(ww*final_emb[i])
        else:
            final_fea = final_emb
        final_fea = [emb.unsqueeze(-1) for emb in final_fea]
        final_fea = torch.cat(final_fea,dim=-1)
        final_fea = torch.mean(final_fea,-1)
        x = self.last_layer(final_fea)
        if segSize is not None:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        else:
            x = nn.functional.log_softmax(x, dim=1)
            return x,clip_emb2
        
       
       
                
        
        
if __name__=='__main__':
    a = torch.rand(2,100,50,60)
    b = torch.rand(2,100,50,60)
    dist = local_pairwise_map(a,b,3)
    print(dist.size())
    print(ofb.size())

# System libs
import os
import argparse
from utils import Evaluator
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from dataset2 import TestDataset_clip,TestDataset_longclip
from models import ModelBuilder, SegmentationModule,ClipWarpNet,NetWarp,ETC,Non_local3d,PropNet,OurWarpMerge,Clip_PSP,ClipOCRNet,NetWarp_ocr,ETC_ocr
from models.td4_psp.td4_psp import td4_psp
from utils import colorEncode, find_recursive, setup_logger,get_common
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
from models.sync_batchnorm.replicate import patch_replication_callback
from collections import OrderedDict
_palette=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
names = {}


def test(segmentation_module, loader, gpu,args,evaluator,eval_video,video):
    segmentation_module.eval()

    gtlist_=[]
    predlist_=[]
    for i,data in enumerate(loader):
        imgs, gts,clip_imgs,_,gtnames = data

        _,_,h,w = imgs.size()
        imgs = imgs.cuda(args.start_gpu)
        gts = gts.cuda(args.start_gpu)
        clip_imgs = [clip_img.cuda(args.start_gpu) for clip_img in clip_imgs]
        batch_data ={}
        batch_data['img_data']= imgs
        batch_data['seg_label'] = gts
        batch_data['clipimgs_data']=clip_imgs
        if args.use_memory:
            if i==0:
                batch_data['is_clean_memory']=True
            else:
                batch_data['is_clean_memory']=False
            
        segSize = (imgs.size(2),
                   imgs.size(3))

        with torch.no_grad():
            if args.method=='tdnet':
                scores = segmentation_module(batch_data,segSize=segSize,pos_id=i%4)
            else:
                scores = segmentation_module(batch_data, segSize=segSize)
            pred = torch.argmax(scores, dim=1)
            pred = pred.data.cpu().numpy()

            
            
            target = gts.squeeze(1).cpu().numpy()


 

            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)
            eval_video.add_batch(target,pred)
            
            ####
            for jj in range(pred.shape[0]):
                imgpred_ = pred[jj]
                target_ = target[jj]
                predlist_.append(imgpred_)
                gtlist_.append(target_)
            ####
            if args.is_save:
                for j in range(pred.shape[0]):
                    imgpred = pred[j]
                    imgpred = Image.fromarray(imgpred.astype('uint8')).convert('P')
                    imgpred.putpalette(_palette)
                    if not os.path.exists(os.path.join(args.saveroot,video)):
                        os.makedirs(os.path.join(args.saveroot,video))
             
                    imgpred.save(os.path.join(args.saveroot,video,gtnames[j].split('.')[0]+'.png'))
             #############
    return gtlist_,predlist_,h,w
def test_all(segmentation_module, loader, gpu,args,evaluator,eval_video,video):
    segmentation_module.eval()

    gtlist_=[]
    predlist_=[]
    target_dic={}
    pred_dic={}
    nn_done=[]
    for i,data in enumerate(loader):
        # process data
        imgs, gts,clip_imgs,clip_targets,gtnames = data
        _,_,h,w = imgs.size()
        imgs = imgs.cuda(args.start_gpu)
        gts = gts.cuda(args.start_gpu)
        clip_imgs = [clip_img.cuda(args.start_gpu) for clip_img in clip_imgs]
        batch_data ={}
        batch_data['clipimgs_data']=clip_imgs
        batch_data['cliplabels_data']=clip_targets
        segSize = (imgs.size(2),
                   imgs.size(3))

        with torch.no_grad():
            scores = segmentation_module(batch_data, segSize=segSize)
            for score,clip_target,gtname in zip(scores,clip_targets,gtnames):
                for ii in range(score.size(0)):
                    ss= score[ii]
                    ll= clip_target[ii]
                    nn = gtname[ii]
                    if nn not in target_dic:
                        if nn not in nn_done:
                            target_dic[nn]=ll
                    if nn not in nn_done:
                        if nn not in pred_dic:
                  
                            pred_dic[nn]=[]
                            pred_dic[nn].append(ss.unsqueeze(0))
               
                        else:
                            pred_dic[nn].append(ss.unsqueeze(0))
                        if len(pred_dic[nn])>args.clip_num-1:
                            tmp = torch.cat(pred_dic[nn],dim=0)
                            tmp = tmp.mean(dim=0,keepdim=True)
                            pred = torch.argmax(tmp, dim=1)
                            pred = pred.data.cpu().numpy()
                            target = target_dic[nn].squeeze(1).cpu().numpy()
                            evaluator.add_batch(target, pred)
                            eval_video.add_batch(target,pred)
                            for jj in range(pred.shape[0]):
                                imgpred_ = pred[jj]
                                target_ = target[jj]
                                predlist_.append(imgpred_)
                                gtlist_.append(target_)
                            del pred_dic[nn]
                            nn_done.append(nn)
                            if args.is_save:
                                for j in range(pred.shape[0]):
                                    imgpred = pred[j]
                                    imgpred = Image.fromarray(imgpred.astype('uint8')).convert('P')
                                    imgpred.putpalette(_palette)
                                    if not os.path.exists(os.path.join(args.saveroot,video)):
                                        os.makedirs(os.path.join(args.saveroot,video))
         
                                    imgpred.save(os.path.join(args.saveroot,video,nn.split('.')[0]+'.png'))
                                    
                        
            
            
    if len(pred_dic)>0:
        for k,v in pred_dic.items():
            tmp = torch.cat(v,dim=0)
            pred_dic[k] = tmp.mean(dim=0,keepdim=True)
                
            
        
        for k,v in pred_dic.items():
            pred = torch.argmax(v, dim=1)
            pred = pred.data.cpu().numpy()
            target = target_dic[k].squeeze(1).cpu().numpy()
    
    
    
    
    
    
        # Add batch sample into evaluator
            evaluator.add_batch(target, pred)
            eval_video.add_batch(target,pred)
        
        ####
            for jj in range(pred.shape[0]):
                imgpred_ = pred[jj]
                target_ = target[jj]
                predlist_.append(imgpred_)
                gtlist_.append(target_)
        ####
            if args.is_save:
                for j in range(pred.shape[0]):
                    imgpred = pred[j]
                    imgpred = Image.fromarray(imgpred.astype('uint8')).convert('P')
                    imgpred.putpalette(_palette)
                    if not os.path.exists(os.path.join(args.saveroot,video)):
                        os.makedirs(os.path.join(args.saveroot,video))
         
                    imgpred.save(os.path.join(args.saveroot,video,k.split('.')[0]+'.png'))
     #############
    return gtlist_,predlist_,h,w



        # visualization


def main(cfg, gpu,args):


    if args.lesslabel:
        num_class = 42
    else:

        num_class = args.num_class
    torch.cuda.set_device(gpu)

    # Network Builders
    if args.method=='tdnet':

        segmentation_module = td4_psp(args=args,backbone='resnet18')
    else:
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights='')
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=num_class,
            weights='',
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        if args.method=='netwarp':
            segmentation_module = NetWarp(
                                      net_encoder, net_decoder, crit,args,cfg.TRAIN.deep_sup_scale
                                     )
        elif args.method=='ETC':
            segmentation_module = ETC(
                                      net_encoder, net_decoder, crit,args,cfg.TRAIN.deep_sup_scale
                                     )
        elif args.method=='nonlocal3d':
            segmentation_module = Non_local3d(args,net_encoder,crit)
           
        elif args.method=='our_warp':
            segmentation_module = ClipWarpNet(net_encoder, net_decoder, crit,args)
        elif args.method=='propnet':
            segmentation_module =PropNet(
                                     net_encoder, net_decoder, crit,args
                                        )

        elif args.method =='our_warp_merge':
            segmentation_module=OurWarpMerge(net_encoder, net_decoder, crit,args)
        elif args.method =='clip_psp':
            segmentation_module=Clip_PSP(net_encoder,crit,args)
        elif args.method=='clip_ocr':
            segmentation_module=ClipOCRNet(net_encoder,crit,args)
        elif args.method=='netwarp_ocr':
            segmentation_module=NetWarp_ocr(net_encoder,crit,args)
        elif args.method=='etc_ocr':
            segmentation_module=ETC_ocr(net_encoder,crit,args)
        
        else:
            raise NotImplementedError

    segmentation_module.cuda(args.start_gpu)


    to_load = torch.load(args.load,map_location=torch.device("cuda:"+str(args.start_gpu)))
    new_state_dict = OrderedDict()
    for k, v in to_load.items():
        name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。

    segmentation_module.load_state_dict(new_state_dict)

    if args.gpu_num>1:
        train_gpu_ = list(range(args.gpu_num))
        train_gpu_ = [int(gpu_+args.start_gpu) for gpu_ in train_gpu_]
        print(train_gpu_)
        segmentation_module = torch.nn.DataParallel(segmentation_module, device_ids=train_gpu_)
        patch_replication_callback(segmentation_module)

    
    with open(os.path.join(args.dataroot,args.split+'.txt')) as f:
            lines=f.readlines()
            videolists = [line[:-1] for line in lines]


    # Dataset and Loader
    evaluator = Evaluator(num_class)
    eval_video = Evaluator(num_class)
    evaluator.reset()
    eval_video.reset()
    total_vmIOU=0.0
    total_vfwIOU=0.0
    total_video = len(videolists)
    total_VC_acc=[]
    for video in videolists:
        eval_video.reset()

        if args.method=='clip_psp' or args.method=='clip_ocr':
            test_dataset=TestDataset_longclip(args.dataroot,video,args,is_train=False)
        else:
            test_dataset = TestDataset_clip(args.dataroot,video,args,is_train=False)
         
        loader_test = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,shuffle=False,num_workers=0,drop_last=False)
        ####
        if args.method=='nonlocal3d':
            gtlist_,predlist_,h,w=test_all(segmentation_module, loader_test, gpu,args,evaluator,eval_video,video)
        else:
            gtlist_,predlist_,h,w=test(segmentation_module, loader_test, gpu,args,evaluator,eval_video,video)
        accs = get_common(gtlist_,predlist_,args.vc_clip_num,h,w)
        print(sum(accs)/len(accs))
        total_VC_acc.extend(accs)
        ####
        v_mIOU =eval_video.Mean_Intersection_over_Union()
        total_vmIOU += v_mIOU
        v_fwIOU = eval_video.Frequency_Weighted_Intersection_over_Union()

        print(video, v_mIOU)
        total_vfwIOU += v_fwIOU

    total_vmIOU  = total_vmIOU/total_video
    total_vfwIOU = total_vfwIOU/total_video

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, video mIOU: {}, video fwIOU: {}".format(Acc, Acc_class, mIoU, FWIoU,total_vmIOU,total_vfwIOU))

    VC_Acc = np.array(total_VC_acc)
    VC_Acc = np.nanmean(VC_Acc)
    print("Video Consistency num :{} acc:{}".format(args.vc_clip_num,VC_Acc))
    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-hrnetv2.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
 
    parser.add_argument("--num_class",type=int,default=124)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--dataroot",type=str,default='')
    parser.add_argument("--saveroot",type=str,default='')
    parser.add_argument("--load_en",type=str,default='')
    parser.add_argument("--load_de",type=str,default='')
    parser.add_argument("--load",type=str,default='')
    parser.add_argument("--batchsize",type=int,default=4)
    parser.add_argument("--split",type=str,default='val')
    parser.add_argument("--is_save",type=str2bool,default=False)
    parser.add_argument("--lesslabel",type=str2bool,default=False)
    parser.add_argument("--use_720p",type=str2bool,default=False)
    parser.add_argument("--clip_num",type=int,default=5)
    parser.add_argument("--dilation_num",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=1)
    parser.add_argument("--propclip2",type=str2bool,default=False)
    parser.add_argument("--early_usecat",type=str2bool,default=False)
    parser.add_argument("--earlyfuse",type=str2bool,default=False)


    ####
    parser.add_argument("--allsup",type=str2bool,default=False)
    parser.add_argument("--allsup_scale",type=float,default=0.3)
    parser.add_argument("--deepsup_scale",type=float,default=0.0)
    parser.add_argument("--linear_combine",type=str2bool,default=False)
    parser.add_argument("--distsoftmax",type=str2bool,default=False)
    parser.add_argument("--distnearest",type=str2bool,default=False)
    parser.add_argument("--temp",type=float,default=3)
    parser.add_argument("--max_distances",type=str,default='10')


    parser.add_argument("--method",type=str,default='',choices=['tdnet','ETC','nonlocal3d','netwarp','our_warp','propnet','our_warp_merge','clip_psp','clip_ocr','netwarp_ocr','etc_ocr'])
    parser.add_argument("--clipocr_all",type=str2bool,default=False)
    parser.add_argument("--dilation2",type=str,default="2,5,9")
    parser.add_argument("--use_memory",type=str2bool,default=False)
    parser.add_argument("--memory_num",type=int,default=8)

    parser.add_argument("--vc_clip_num",type=int,default=8)
    parser.add_argument("--psp_weight",type=str2bool,default=False)
    ####
    args = parser.parse_args()

    args.max_distances = args.max_distances.split(',')
    args.max_distances = [int(dd) for dd in args.max_distances]
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    cfg.MODEL.weights_encoder = args.load_en
    cfg.MODEL.weights_decoder = args.load_de


    if not os.path.isdir(args.saveroot):
        os.makedirs(args.saveroot)

    main(cfg, args.start_gpu,args)
    print(args)

# System libs
import os
import time
import json
# import math
import random
import argparse
from utils import Evaluator
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from config import cfg
from dataset import TrainDataset
from dataset2 import BaseDataset, BaseDataset_clip,TestDataset_clip,BaseDataset_longclip
from models import ModelBuilder, ClipWarpNet,NetWarp,ETC,Non_local3d,PropNet,OurWarpMerge,Clip_PSP,ClipOCRNet,NetWarp_ocr,ETC_ocr
from models.td4_psp.td4_psp import td4_psp
from models.td4_psp.loss import OhemCELoss2D
from utils import AverageMeter, parse_devices, setup_logger
from models.sync_batchnorm.replicate import patch_replication_callback
import numpy as np


# train one epoch
def train(segmentation_module, data_loader, optimizers, history, epoch, cfg,args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    epoch_iters = len(data_loader)
    max_iters = epoch_iters * cfg.TRAIN.num_epoch


    # main loop
    tic = time.time()
    it_=0
    for i,data in enumerate(data_loader):
        it_+=1
        #continue
        # load a batch of data
        clip_imgs, clip_gts = data
        clip_imgs = [imgs.cuda(args.start_gpu) for imgs in clip_imgs]
        clip_gts = [gts.cuda(args.start_gpu) for gts in clip_gts]
        batch_data ={}
        if args.clip_num%2==0:
            idx = args.clip_num/2
        else:
            idx = (args.clip_num-1)/2
        #print(len(clip_imgs))
        if args.method=='tdnet':
            assert args.clip_num==4
        #    assert args.dilation_num==0
            batch_data['clipimgs_data'] = clip_imgs
            batch_data['cliplabels_data'] = clip_gts
        elif args.method =='nonlocal3d':
            batch_data['clipimgs_data'] = clip_imgs
            batch_data['cliplabels_data'] = clip_gts
        
        elif args.method=='netwarp' or args.method=='ETC' or args.method=='netwarp_ocr' or args.method=='etc_ocr':
            assert args.clip_num==2
            assert args.dilation_num==0
            batch_data['img_data']= clip_imgs.pop(int(idx))
            batch_data['seg_label'] = clip_gts.pop(int(idx))
            batch_data['clipimgs_data'] = clip_imgs
            batch_data['cliplabels_data'] = clip_gts
        elif args.method=='our_warp' or args.method=='propnet' or args.method=='our_warp_merge':
            batch_data['img_data']= clip_imgs.pop(int(idx))
            batch_data['seg_label'] = clip_gts.pop(int(idx))
            batch_data['clipimgs_data'] = clip_imgs
            batch_data['cliplabels_data'] = clip_gts
        elif args.method=='clip_psp' or args.method=='clip_ocr':
            batch_data['img_data']= clip_imgs[0]
            batch_data['seg_label'] = clip_gts[0]
            batch_data['clipimgs_data'] = clip_imgs[1:]
            batch_data['cliplabels_data'] = clip_gts[1:]
    
        else:
            raise(NotImplementedError)
        batch_data['step']=it_
        
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg,max_iters,args)

        # forward pass
        if args.method =='tdnet':
            loss, acc = segmentation_module(batch_data,pos_id=it_%4)

        else:
            loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
#        for optimizer in optimizers:
        optimizers.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.data.item())
        history['train']['acc'].append(acc.data.item())

def test(segmentation_module, args=None):

    label_num_ = args.num_class
    segmentation_module.eval()
    evaluator = Evaluator(label_num_)

    print('validation')
    
    with open(os.path.join(args.dataroot,'val.txt'),'r') as f:
        lines=f.readlines()
        videolists = [line[:-1] for line in lines]

    for video in videolists:
        test_dataset = TestDataset_clip(args.dataroot,video,args,is_train=True)
        loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,shuffle=False,num_workers=args.workers,drop_last=False)

        for i,data in enumerate(loader):
            # process data
            print('[{}]/[{}]'.format(i,len(loader)))
            imgs, gts,clip_imgs,_,_ = data
            imgs = imgs.cuda(args.start_gpu)
            gts = gts.cuda(args.start_gpu)
            clip_imgs = [img.cuda(args.start_gpu) for img in clip_imgs]
            batch_data ={}
            batch_data['img_data']= imgs
            batch_data['seg_label'] = gts
            batch_data['clipimgs_data']=clip_imgs
            segSize = (imgs.size(2),
                       imgs.size(3))
    
            with torch.no_grad():
                 
                scores = segmentation_module(batch_data, segSize=segSize)
                pred = torch.argmax(scores, dim=1)
                pred = pred.data.cpu().numpy()
                target = gts.squeeze(1).cpu().numpy()
                 
                # Add batch sample into evaluator
                evaluator.add_batch(target, pred)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU =evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))






def checkpoint(opt,nets, history, args, epoch):
    print('Saving checkpoints...')

    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot)
    torch.save(
        nets.state_dict(),
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))
    torch.save(
        opt.state_dict(),
        '{}/opt_epoch_{}.pth'.format(args.saveroot, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(model, cfg,args):

    if args.fix:
        model.get_1x_lr_params()
        model.get_1x_lr_params_bias()
        train_params = [{'params': model.get_10x_lr_params(), 'lr': args.lr ,'weight_decay':cfg.TRAIN.weight_decay},
                          {'params':model.get_10x_lr_params_bias(),'lr':args.lr,'weight_decay':0}]

    else:

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr*0.1,'weight_decay':cfg.TRAIN.weight_decay},
                           {'params': model.get_10x_lr_params(), 'lr': args.lr ,'weight_decay':cfg.TRAIN.weight_decay},
                            {'params':model.get_1x_lr_params_bias(),'lr':args.lr*0.1,'weight_decay':0},
                            {'params':model.get_10x_lr_params_bias(),'lr':args.lr,'weight_decay':0} ]
    
    
    optimizer = torch.optim.SGD(
        train_params,
        lr=args.lr,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return optimizer


def adjust_learning_rate(optimizer, cur_iter, cfg,max_iters,args):
    scale_running_lr = ((1. - float(cur_iter) / max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = args.lr * scale_running_lr

    if args.fix:
        optimizer.param_groups[0]['lr'] = cfg.TRAIN.running_lr_encoder
        optimizer.param_groups[1]['lr'] = cfg.TRAIN.running_lr_encoder

    else:

        optimizer.param_groups[0]['lr'] = cfg.TRAIN.running_lr_encoder*0.1
        optimizer.param_groups[1]['lr'] = cfg.TRAIN.running_lr_encoder
        optimizer.param_groups[2]['lr'] = cfg.TRAIN.running_lr_encoder*0.1
        optimizer.param_groups[3]['lr'] = cfg.TRAIN.running_lr_encoder





def main(cfg, gpus):
    # Network Builders


    label_num_=args.num_class

    if args.method=='tdnet':
        n_img_per_gpu = int(args.batchsize/args.gpu_num)
        n_min = n_img_per_gpu * args.cropsize * args.cropsize // 16
        loss_fn=OhemCELoss2D(thresh=0.7,n_min=n_min,ignore_index=255)
        segmentation_module = td4_psp(args=args,backbone='resnet18',loss_fn=loss_fn)
        segmentation_module.pretrained_init()

    else:
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder,args=args)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            num_class=label_num_,
            weights=cfg.MODEL.weights_decoder)

        crit = nn.NLLLoss(ignore_index=255)

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

            if args.deepsup_scale>0.:
                segmentation_module = ClipWarpNet(
                    net_encoder, net_decoder, crit,args,args.deepsup_scale)
            else: 
            
                segmentation_module = ClipWarpNet(
                    net_encoder, net_decoder, crit,args)
        elif args.method=='propnet':
            segmentation_module=PropNet(
                                     net_encoder, net_decoder, crit,args,deep_sup_scale=args.deepsup_scale
                                        )
        elif args.method =='our_warp_merge':
            segmentation_module=OurWarpMerge(net_encoder, net_decoder, crit,args,deep_sup_scale=0.4)
        
        elif args.method =='clip_psp':
            segmentation_module=Clip_PSP(net_encoder,crit,args,deep_sup_scale=0.4)
        elif args.method =='clip_ocr':
            segmentation_module=ClipOCRNet(net_encoder,crit,args,deep_sup_scale=0.4)
        elif args.method=='netwarp_ocr':
            segmentation_module=NetWarp_ocr(net_encoder,crit,args,deep_sup_scale=0.4)
        elif args.method=='etc_ocr':
            segmentation_module=ETC_ocr(net_encoder,crit,args,deep_sup_scale=0.4)
        else:
            raise(NotImplementedError)

    # Dataset and Loader
    if args.method=='clip_psp' or args.method=='clip_ocr':
        dataset_train = BaseDataset_longclip(args,'train')
    else:
        dataset_train = BaseDataset_clip(
             args,
            'train'
            )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batchsize, # we have modified data_parallel
        shuffle=True,  # we do not use this param
        num_workers=args.workers,
        drop_last=True,
        pin_memory=False)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    

    # load nets into gpu
    
    segmentation_module.cuda(args.start_gpu)
    optimizer = create_optimizers(segmentation_module, cfg,args)
    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join('./resume','model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(args.start_gpu)))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in to_load.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        cfg.TRAIN.start_epoch=args.resume_epoch
        segmentation_module.load_state_dict(new_state_dict)
        optimizer.load_state_dict(torch.load(os.path.join('./resume','opt_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(args.start_gpu))))
        print('resume from epoch {}'.format(args.resume_epoch))

    if args.gpu_num>1:
        train_gpu_ = list(range(args.gpu_num))
        train_gpu_ = [int(gpu_+args.start_gpu) for gpu_ in train_gpu_]
        print(train_gpu_)
        segmentation_module = torch.nn.DataParallel(segmentation_module, device_ids=train_gpu_)
        patch_replication_callback(segmentation_module)

#    print(segmentation_module)
    # Set up optimizers

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    #if len(args.resume_dir)>0:
    #    resume_epoch = args.resume_dir.split('.')[]
    


    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        print('Epoch {}'.format(epoch))
        #checkpoint(optimizer,segmentation_module, history, args, epoch+1)
        train(segmentation_module, loader_train, optimizer, history, epoch+1, cfg,args)

###################        # checkpointing
        if (epoch+1)%20==0:
            checkpoint(optimizer,segmentation_module, history, args, epoch+1)
            if args.validation:
                test(segmentation_module,args)
#
    print('Training Done!')


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
    "--predir",
    default= '../../ade20k-hrnetv2-c1'
    )
    parser.add_argument("--num_class",type=int,default=124)
    parser.add_argument("--batchsize",type=int,default=16)
    parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=1)
    parser.add_argument("--dataroot",type=str,default='')
    parser.add_argument("--trainfps",type=int,default=1)
    parser.add_argument("--lr",type=float,default=0.02)
    parser.add_argument("--multi_scale",type=str2bool,default=False)
    parser.add_argument("--saveroot",type=str,default='')
    parser.add_argument("--totalepoch",type=int,default=30)
    parser.add_argument("--dataroot2",type=str,default='')
    parser.add_argument("--usetwodata",type=str2bool,default=False)
    parser.add_argument("--cropsize",type=int,default=531)
    parser.add_argument("--validation",type=str2bool,default=True)
    parser.add_argument("--lesslabel",type=str2bool,default=False)
    parser.add_argument("--clip_num",type=int,default=5)
    parser.add_argument("--dilation_num",type=int,default=3)
    
    parser.add_argument("--clip_up",type=str2bool,default=False)
    parser.add_argument("--clip_middle",type=str2bool,default=False)

    parser.add_argument("--fix",type=str2bool,default=False)
    parser.add_argument("--othergt",type=str2bool,default=False)
    parser.add_argument("--propclip2",type=str2bool,default=False)
    parser.add_argument("--early_usecat",type=str2bool,default=False)
    parser.add_argument("--earlyfuse",type=str2bool,default=False)

    parser.add_argument("--weight_decay",type=float,default=1e-4)
    #####

    ####
    parser.add_argument("--allsup",type=str2bool,default=False)
    parser.add_argument("--allsup_scale",type=float,default=0.3)
    parser.add_argument("--deepsup_scale",type=float,default=0.4)
    parser.add_argument("--linear_combine",type=str2bool,default=False)
    parser.add_argument("--distsoftmax",type=str2bool,default=False)
    parser.add_argument("--distnearest",type=str2bool,default=False)
    parser.add_argument("--temp",type=float,default=3)
    parser.add_argument("--max_distances",type=str,default='10')

    
    parser.add_argument("--pre_enc",type=str,default='')
    parser.add_argument("--pre_dec",type=str,default='')

    ####
    parser.add_argument("--method",type=str,default='',choices=['netwarp','ETC','nonlocal3d','tdnet','our_warp','propnet','our_warp_merge','clip_psp','clip_ocr','netwarp_ocr','etc_ocr'])
    
    parser.add_argument("--dilation2",type=str,default="2,5,9")
    parser.add_argument("--resume_epoch",type=int,default=0)


    parser.add_argument("--clipocr_all",type=str2bool,default=False)
    parser.add_argument("--use_memory",type=str2bool,default=False)
    parser.add_argument("--memory_num",type=int,default=8)
    parser.add_argument("--st_weight",type=float,default=0.1)
    parser.add_argument("--psp_weight",type=str2bool,default=False)
    ####


    #####
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    args.max_distances = args.max_distances.split(',')
    args.max_distances = [int(dd) for dd in args.max_distances]


    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    cfg.MODEL.weights_encoder = args.pre_enc
    cfg.MODEL.weights_decoder = args.pre_dec

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.num_epoch = args.totalepoch

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

    cfg.TRAIN.weight_decay=args.weight_decay

    cfg.TRAIN.lr_encoder = args.lr
    cfg.TRAIN.lr_decoder = args.lr
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    print(args)

    main(cfg, gpus)

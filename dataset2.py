import os
import random
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def dilation_list(list_,num):
    newlist=[]
    a = np.random.choice(list(range(num)))

    for k in range(len(list_)):
        if k%(num+1)==a:
            newlist.append(list_[k])
    return newlist



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot,video,args):
        # parse options
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        self.dataroot = dataroot
        self.video = video
        self.imglist=[]
        self.args = args


        v_path = os.path.join(self.dataroot,'data',video,'origin')
        imglist = sorted(os.listdir(v_path))
        
        self.imglist = imglist 
        
        ###xo##
        #new_list = []
        #for k in range(len(imglist)):
        #    if k%15==0:
        #        new_list.append(imglist[k])
        #self.imglist = new_list
            


        #####
        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
#        image = self.pad_image(image, h, w, self.crop_size,
#                                (0.0, 0.0, 0.0))
#        label = self.pad_image(label, h, w, self.crop_size,
#                                (self.ignore_label,))

#        new_h, new_w = label.shape
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    #def segm_transform(self, segm):
    #    # to tensor, -1 to 149
    #    segm = torch.from_numpy(segm).long() 
    #    return segm
    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self,idx):
        imgname = self.imglist[idx]
        img =  Image.open(os.path.join(self.dataroot,'data',self.video,'origin',imgname)).convert('RGB')
        gtname = imgname.split('.')[0]+'.png'
        if self.args.lesslabel:
            gt = Image.open(os.path.join(self.dataroot,'data',self.video,'mask_42label',gtname))
  
        else:
            gt = Image.open(os.path.join(self.dataroot,'data',self.video,'mask',gtname))
        if self.args.use_720p:
            img = img.resize((1080,720),Image.BILINEAR)
            gt = gt.resize((1080,720),Image.NEAREST)



            # random_flip
        img = np.float32(np.array(img)) / 255.
        img = self.img_transform(img)
        gt = np.array(gt)
        gt = self.segm_transform(gt)
        return img,gt,gtname

def dilation_lists(list_,num):
    newlists=[]
    for a in range(num+1):
        newlist=[]
        for k in range(len(list_)):
            if k%(num+1)==a:
                newlist.append(list_[k])
        newlists.append(newlist)
    return newlists
########

class TestDataset_clip(torch.utils.data.Dataset):
    def __init__(self, dataroot,video,args,is_train=False):
        # parse options
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        self.dataroot = dataroot
        self.video = video
        self.clip_num = args.clip_num
        self.dilation = args.dilation_num
        self.imglist=[]
        self.args = args

        self.is_train =is_train

        v_path = os.path.join(self.dataroot,'data',video,'origin')
        imglist = sorted(os.listdir(v_path))
        
        self.imglist = imglist 
        self.dilists = dilation_lists(self.imglist,self.dilation)
        self.imglist2 = []

        if self.is_train:
            for k in range(len(imglist)):
                if k%15==0:
                    self.imglist2.append(imglist[k])
    


        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
#        image = self.pad_image(image, h, w, self.crop_size,
#                                (0.0, 0.0, 0.0))
#        label = self.pad_image(label, h, w, self.crop_size,
#                                (self.ignore_label,))

#        new_h, new_w = label.shape
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    #def segm_transform(self, segm):
    #    # to tensor, -1 to 149
    #    segm = torch.from_numpy(segm).long() 
    #    return segm
    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        if self.is_train:
            return len(self.imglist2)
        else:

            return len(self.imglist)
    
    def __getitem__(self,index):
        if self.is_train:
            img = self.imglist2[index]
        else:
            img = self.imglist[index]
        if self.args.method=='nonlocal3d':
            imagenames=[]

        else:
            imagenames=img
#        imagenames=[]
        #####
#        imagenames.append(img)
        _img = Image.open(os.path.join(self.dataroot,'data',self.video,'origin',img))
        if self.args.lesslabel:
            _target = Image.open(os.path.join(self.dataroot,'data',self.video,'mask_42label',img.split('.')[0]+'.png'))

        else:
            _target = Image.open(os.path.join(self.dataroot,'data',self.video,'mask',img.split('.')[0]+'.png'))

        _img = np.float32(np.array(_img)) / 255.
        _img = self.img_transform(_img)
        _target = np.array(_target)
        _target = self.segm_transform(_target)

        for dilist in self.dilists:
            if img in dilist:
                thelist = dilist
        imgindex = thelist.index(img)
        if self.clip_num%2==0:
            add = self.clip_num/2
        else:
            add = (self.clip_num-1)/2
        add = int(add)
        addleft = add
        if self.clip_num%2==0:
            addright = add-1
        else:
            addright = add
        if imgindex-addleft<0:
            start = 0
            end = start+self.clip_num
            if end>=len(thelist):
                end = len(thelist)
 
        elif imgindex+addright>=len(thelist):
            end = len(thelist)
            start = end-self.clip_num
            if start<0:
                start=0
        else:
            start = imgindex-addleft
            end = start + self.clip_num


        clips_img = []
        clips_target=[]
        if end-start<2:
            clips_img.append(_img)    
            clips_target.append(_target)
        else:
            for idx in range(start,end):
                if self.args.method=='nonlocal3d':
                    pass
                else:
                    if idx == imgindex:
                        continue
                imgname = thelist[idx]
#                imagenames.append(imgname)
                if self.args.method=='nonlocal3d':
                    imagenames.append(imgname)
                img_ = Image.open(os.path.join(self.dataroot,'data',self.video,'origin',imgname))


                if self.args.lesslabel:
                    target_ = Image.open(os.path.join(self.dataroot,'data',self.video,'mask_42label',imgname.split('.')[0]+'.png'))

                else:
                    target_ = Image.open(os.path.join(self.dataroot,'data',self.video,'mask',imgname.split('.')[0]+'.png'))
                img_ = np.float32(np.array(img_)) / 255.
                img_ = self.img_transform(img_)
                target_ = np.array(target_)
                target_ = self.segm_transform(target_)
                clips_img.append(img_)
                clips_target.append(target_)
        
#        print(imagenames)
        if self.is_train:
            return _img,_target,clips_img,clips_target    ,imagenames

        else:
            return _img,_target,clips_img,clips_target,imagenames
########




class TestDataset_longclip(torch.utils.data.Dataset):
    def __init__(self, dataroot,video,args,is_train=False):
        # parse options
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        self.dataroot = dataroot
        self.video = video
        self.clip_num = args.clip_num
        self.dilation = args.dilation2
        self.dilation = self.dilation.split(',')
        self.dilation = [int(dil) for dil in self.dilation] 
        assert(len(self.dilation)+1==self.clip_num)
        self.imglist=[]
        self.args = args

        self.is_train =is_train

        v_path = os.path.join(self.dataroot,'data',video,'origin')
        imglist = sorted(os.listdir(v_path))
        
        
        self.imglist = imglist 
        #print(self.imglist)
        #self.dilists = dilation_lists(self.imglist,self.dilation)
        self.imglist2 = []

        if self.is_train:
            for k in range(len(imglist)):
                if k%15==0:
                    self.imglist2.append(imglist[k])
    


        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
#        image = self.pad_image(image, h, w, self.crop_size,
#                                (0.0, 0.0, 0.0))
#        label = self.pad_image(label, h, w, self.crop_size,
#                                (self.ignore_label,))

#        new_h, new_w = label.shape
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    #def segm_transform(self, segm):
    #    # to tensor, -1 to 149
    #    segm = torch.from_numpy(segm).long() 
    #    return segm
    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        if self.is_train:
            return len(self.imglist2)
        else:

            return len(self.imglist)
    
    def __getitem__(self,index):
        img = self.imglist[index]
        imagenames=img


#        imagenames=[]
        #####
#        imagenames.append(img)
        _img = Image.open(os.path.join(self.dataroot,'data',self.video,'origin',img))
        if self.args.lesslabel:
            _target = Image.open(os.path.join(self.dataroot,'data',self.video,'mask_42label',img.split('.')[0]+'.png'))

        else:
            _target = Image.open(os.path.join(self.dataroot,'data',self.video,'mask',img.split('.')[0]+'.png'))

        _img = np.float32(np.array(_img)) / 255.
        _img = self.img_transform(_img)
        _target = np.array(_target)
        _target = self.segm_transform(_target)



        clips_img=[]
        clips_target=[]
        for dil in self.dilation:
            if index+self.dilation[-1]>=len(self.imglist):
                idx = index-dil
            else:
                idx = index+dil

            imgname = self.imglist[idx]
        #    imagenames.append(imgname)
            img_ = Image.open(os.path.join(self.dataroot,'data',self.video,'origin',imgname))


            if self.args.lesslabel:
                target_ = Image.open(os.path.join(self.dataroot,'data',self.video,'mask_42label',imgname.split('.')[0]+'.png'))

            else:
                target_ = Image.open(os.path.join(self.dataroot,'data',self.video,'mask',imgname.split('.')[0]+'.png'))
            img_ = np.float32(np.array(img_)) / 255.
            img_ = self.img_transform(img_)
            target_ = np.array(target_)
            target_ = self.segm_transform(target_)
            clips_img.append(img_)
            clips_target.append(target_)
        
#        print(imagenames)
        return _img,_target,clips_img,clips_target,imagenames

########

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,args,split='train'):
        # parse options
        if args.train_filter:
            self.cropsize = (480,720)
        
        else:
            self.cropsize = (args.cropsize,args.cropsize)
        self.dataroot = args.dataroot
        self.trainfps = args.trainfps
        self.split = split
        with open(os.path.join(self.dataroot,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        self.args = args
        self.scale = [0.8,1.,1.5,2.0]
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list

        if self.split=='val':
            self.trainfps=1
        self.imglist=[]
        for video in self.videolists:
            v_path = os.path.join(self.dataroot,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))

            

            num = int(15./self.trainfps)
            for k in range(len(imglist)):
                if k%num==0:
                    self.imglist.append((video,imglist[k]))
        

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def multi_rand_crop(self, image, label,origin_hw):
        h, w = image.shape[:-1]
        orih,oriw = origin_hw
        if orih<=1080 or oriw<=1920:
            if orih!=h or oriw!=w:
                x = random.randint(0, w  - oriw)

                y = random.randint(0, h - orih)
                image = image[y:y+orih, x:x+oriw]

                label = label[y:y+orih, x:x+oriw]
                return image,label

            else:
                return image,label
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        if h<w:
            short_size = h
        else:
            short_size =w

        
        padw = self.cropsize[1] - w if w < self.cropsize[1] else 0
        padh = self.cropsize[0] - h if h < self.cropsize[0] else 0
        
        label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
        image = np.pad(image,((padh,padh),(padw,padw),(0,0)),'constant')

        h, w  = image.shape[:-1]




        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self,idx):
        video,img  = self.imglist[idx]
        _img = Image.open(os.path.join(self.dataroot,'data',video,'origin',img))


        if self.args.lesslabel:
            _target = Image.open(os.path.join(self.dataroot,'data',video,'mask_42label',img.split('.')[0]+'.png'))


        else:
            _target = Image.open(os.path.join(self.dataroot,'data',video,'mask',img.split('.')[0]+'.png'))
        img = _img.convert('RGB')
        segm = _target
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])
            # random_flip
        if self.split=='train':
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            if self.args.multi_scale:
                scale = np.random.choice(self.scale)
                w,h = img.size
                if scale != 1.:
                    new_w = int(w*scale)
                    new_h = int(h*scale)
                    img = img.resize((new_w,new_h),Image.BILINEAR)
                    segm = segm.resize((new_w,new_h),Image.NEAREST)
        img = np.float32(np.array(img)) / 255.
        segm = np.array(segm)
        if self.split=='train':
            img, segm = self.rand_crop(img,segm)

        img = self.img_transform(img)
        segm = self.segm_transform(segm)
        
        return img, segm
        

class BaseDataset_clip(torch.utils.data.Dataset):
    def __init__(self,args,split='train'):
        # parse options
        self.cropsize = (args.cropsize,args.cropsize)
        self.dataroot = args.dataroot
        self.trainfps = args.trainfps
        self.clipnum = args.clip_num
        self.dilation = args.dilation_num
        self.split = split
        with open(os.path.join(self.dataroot,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        self.args = args
        self.scale = [0.8,1.,1.5,2.0]
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list

#        if self.split=='val':
#            self.trainfps=1
        self.imgdic={}
        for video in self.videolists:
            v_path = os.path.join(self.dataroot,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            self.imgdic[video]=imglist

            

#            num = int(15./self.trainfps)
#            for k in range(len(imglist)):
#                if k%num==0:
#                    self.imglist.append((video,imglist[k]))
        

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def multi_rand_crop(self, image, label,origin_hw):
        h, w = image.shape[:-1]
        orih,oriw = origin_hw
        if orih<=1080 or oriw<=1920:
            if orih!=h or oriw!=w:
                x = random.randint(0, w  - oriw)

                y = random.randint(0, h - orih)
                image = image[y:y+orih, x:x+oriw]

                label = label[y:y+orih, x:x+oriw]
                return image,label

            else:
                return image,label
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def rand_crop(self, images, labels):
        h, w = images[0].shape[:-1]

        
        padw = self.cropsize[1] - w if w < self.cropsize[1] else 0
        padh = self.cropsize[0] - h if h < self.cropsize[0] else 0
        
#        label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
        image_ = np.pad(images[0],((padh,padh),(padw,padw),(0,0)),'constant')

        h, w  = image_.shape[:-1]
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        images_=[]
        labels_=[]
        for image,label in zip(images,labels):
            label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
            image = np.pad(image,((padh,padh),(padw,padw),(0,0)),'constant')
        
        


            image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
            label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]
            images_.append(image)
            labels_.append(label)

        return images_, labels_
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.videolists)
    
    def __getitem__(self,idx):
        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        imglists = dilation_lists(imglist,self.dilation)
        for step in range(10):
            idd = np.random.choice(list(range(len(imglists))))
            imglist = imglists[idd]
            if len(imglist)>self.clipnum:
                break
            
        
        if len(imglist)<=self.clipnum:
            for ii in range(self.clipnum+1-len(imglist)):
                imglist.append(imglist[-1])
        imagenames = []

        imgidxs = list(range(len(imglist)))
        imgidxs_ = imgidxs[:-self.clipnum]
        imgid = np.random.choice(imgidxs_,1)
        imgid = imgid[0]
        clips_img = []
        clips_target=[]


        Flip_flag = np.random.choice([0,1])
        scale = np.random.choice(self.scale)
        clips_i=[]
        clips_t=[]
        for i in range(imgid,imgid+self.clipnum):
            imgname = imglist[i]
            img_ = Image.open(os.path.join(self.dataroot,'data',video,'origin',imgname)).convert('RGB')
            if self.args.lesslabel:
                target_ = Image.open(os.path.join(self.dataroot,'data',video,'mask_42label',imgname.split('.')[0]+'.png'))

            else:
                target_ = Image.open(os.path.join(self.dataroot,'data',video,'mask',imgname.split('.')[0]+'.png'))
            segm = target_
            if self.split=='train':
                if Flip_flag:
                    img_ = img_.transpose(Image.FLIP_LEFT_RIGHT)
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                if self.args.multi_scale:
                    scale = scale
                    w,h = img_.size
                    if scale != 1.:
                        new_w = int(w*scale)
                        new_h = int(h*scale)
                        img_ = img_.resize((new_w,new_h),Image.BILINEAR)
                        segm = segm.resize((new_w,new_h),Image.NEAREST)
            img_ = np.float32(np.array(img_)) / 255.
            segm = np.array(segm)
            clips_i.append(img_)
            clips_t.append(segm)
            imagenames.append(imgname)
            
        if self.split=='train':
            clips_i, clips_t = self.rand_crop(clips_i,clips_t)

        for img_,segm in zip(clips_i,clips_t):
            img_ = self.img_transform(img_)
            segm = self.segm_transform(segm)
            clips_img.append(img_)
            clips_target.append(segm)

            # random_flip
        #print(imagenames)
        return clips_img, clips_target


class BaseDataset_longclip(torch.utils.data.Dataset):
    def __init__(self,args,split='train'):
        # parse options
        self.cropsize = (args.cropsize,args.cropsize)
        self.dataroot = args.dataroot
        self.trainfps = args.trainfps
        self.clipnum = args.clip_num
        self.dilation = args.dilation2
        self.dilation = self.dilation.split(',')
        self.dilation = [int(dil) for dil in self.dilation]
        
        ####
#        self.dilation = [2,5,9]
        assert len(self.dilation)+1== self.clipnum

        ####
        self.split = split
        with open(os.path.join(self.dataroot,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        self.args = args
        self.scale = [0.8,1.,1.5,2.0]
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list

#        if self.split=='val':
#            self.trainfps=1
        self.imgdic={}
        for video in self.videolists:
            v_path = os.path.join(self.dataroot,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            self.imgdic[video]=imglist

            

#            num = int(15./self.trainfps)
#            for k in range(len(imglist)):
#                if k%num==0:
#                    self.imglist.append((video,imglist[k]))
        

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def multi_rand_crop(self, image, label,origin_hw):
        h, w = image.shape[:-1]
        orih,oriw = origin_hw
        if orih<=1080 or oriw<=1920:
            if orih!=h or oriw!=w:
                x = random.randint(0, w  - oriw)

                y = random.randint(0, h - orih)
                image = image[y:y+orih, x:x+oriw]

                label = label[y:y+orih, x:x+oriw]
                return image,label

            else:
                return image,label
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def rand_crop(self, images, labels):
        h, w = images[0].shape[:-1]

        
        padw = self.cropsize[1] - w if w < self.cropsize[1] else 0
        padh = self.cropsize[0] - h if h < self.cropsize[0] else 0
        
#        label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
        image_ = np.pad(images[0],((padh,padh),(padw,padw),(0,0)),'constant')

        h, w  = image_.shape[:-1]
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        images_=[]
        labels_=[]
        for image,label in zip(images,labels):
            label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
            image = np.pad(image,((padh,padh),(padw,padw),(0,0)),'constant')
        
        


            image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
            label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]
            images_.append(image)
            labels_.append(label)

        return images_, labels_
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm[segm==0]=255
        segm = segm-1
        segm[segm==254]=255
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.videolists)
    
    def __getitem__(self,idx):
        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        if np.random.random()<0.5:
            imglist=imglist[::-1]
        imglist_s = imglist[:-self.dilation[-1]]
        while len(imglist_s)<1:
            imglist.append(imglist[-1])
            imglist_s = imglist[:-self.dilation[-1]]
        
        idx = np.random.choice(list(range(len(imglist_s))))
        this_step=[idx]
        for dil in self.dilation:
            this_step.append(idx+dil)
       
        
        
        clips_img = []
        clips_target=[]


        Flip_flag = np.random.choice([0,1])
        scale = np.random.choice(self.scale)
        clips_i=[]
        clips_t=[]

#        imagenames=[]
        for i in this_step:
            imgname = imglist[i]
            #imagenames.append(imgname)
            img_ = Image.open(os.path.join(self.dataroot,'data',video,'origin',imgname)).convert('RGB')
            target_ = Image.open(os.path.join(self.dataroot,'data',video,'mask',imgname.split('.')[0]+'.png'))
            segm = target_
            if self.split=='train':
                if Flip_flag:
                    img_ = img_.transpose(Image.FLIP_LEFT_RIGHT)
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                if self.args.multi_scale:
                    scale = scale
                    w,h = img_.size
                    if scale != 1.:
                        new_w = int(w*scale)
                        new_h = int(h*scale)
                        img_ = img_.resize((new_w,new_h),Image.BILINEAR)
                        segm = segm.resize((new_w,new_h),Image.NEAREST)
            img_ = np.float32(np.array(img_)) / 255.
            segm = np.array(segm)
            clips_i.append(img_)
            clips_t.append(segm)
#            imagenames.append(imgname)
            
        if self.split=='train':
            clips_i, clips_t = self.rand_crop(clips_i,clips_t)

        for img_,segm in zip(clips_i,clips_t):
            img_ = self.img_transform(img_)
            segm = self.segm_transform(segm)
            clips_img.append(img_)
            clips_target.append(segm)

        #print(imagenames)
            # random_flip
        return clips_img, clips_target



class TwoDataset(torch.utils.data.Dataset):
    def __init__(self,args,split='train'):
        # parse options
        self.cropsize = (args.cropsize,args.cropsize)
        self.dataroot = args.dataroot
        self.dataroot2 = args.dataroot2
        self.trainfps = args.trainfps
        self.split = split
        with open(os.path.join(self.dataroot,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        self.args = args
        self.scale = [0.8,1.,1.5]
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list

        if self.split=='val':
            self.trainfps=1
        self.imglist=[]
        for video in self.videolists:
            v_path = os.path.join(self.dataroot,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))

            

            num = int(15./self.trainfps)
            for k in range(len(imglist)):
                if k%num==0:
                    self.imglist.append((video,imglist[k]))
        

        self.imglist2=sorted(os.listdir(os.path.join(self.dataroot2,'origin')))
        


        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def multi_rand_crop(self, image, label,origin_hw):
        h, w = image.shape[:-1]
        orih,oriw = origin_hw
        if orih<=1080 or oriw<=1920:
            if orih!=h or oriw!=w:
                x = random.randint(0, w  - oriw)

                y = random.randint(0, h - orih)
                image = image[y:y+orih, x:x+oriw]

                label = label[y:y+orih, x:x+oriw]
                return image,label

            else:
                return image,label
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        if h<w:
            short_size = h
        else:
            short_size =w

        
        padw = self.cropsize[1] - w if w < self.cropsize[1] else 0
        padh = self.cropsize[0] - h if h < self.cropsize[0] else 0
        
        label = np.pad(label,((padh,padh),(padw,padw)),'constant', constant_values=(255,255))
        image = np.pad(image,((padh,padh),(padw,padw),(0,0)),'constant')

        h, w  = image.shape[:-1]




        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self,idx):
        video,img  = self.imglist[idx]
        _img = Image.open(os.path.join(self.dataroot,'data',video,'origin',img))

        if self.args.lesslabel:
            _target = Image.open(os.path.join(self.dataroot,'data',video,'mask_42label',img.split('.')[0]+'.png'))


        else:

            _target = Image.open(os.path.join(self.dataroot,'data',video,'mask',img.split('.')[0]+'.png'))
        img = _img.convert('RGB')
        segm = _target
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

            # random_flip
        if self.split=='train':
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            if self.args.multi_scale:
                scale = np.random.choice(self.scale)
                w,h = img.size
                if scale != 1.:
                    new_w = int(w*scale)
                    new_h = int(h*scale)
                    img = img.resize((new_w,new_h),Image.BILINEAR)
                    segm = segm.resize((new_w,new_h),Image.NEAREST)
        img = np.float32(np.array(img)) / 255.
        segm = np.array(segm)
        if self.split=='train':
            img, segm = self.rand_crop(img,segm)

        img = self.img_transform(img)
        segm = self.segm_transform(segm)


        ############
        imgname2 = np.random.choice(self.imglist2)
        imgname2 = imgname2
     
        img2 = Image.open(os.path.join(self.dataroot2,'origin',imgname2)).convert('RGB')
        target2 = Image.open(os.path.join(self.dataroot2,'mask',imgname2.split('.')[0]+'.png'))
        assert(img2.size[0]==target2.size[0])
        assert(img2.size[1]==target2.size[1])

        if np.random.choice([0, 1]):
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            target2 = target2.transpose(Image.FLIP_LEFT_RIGHT)
        if self.args.multi_scale:
            scale = np.random.choice([1.,1.5,2.])
            w,h = img2.size
            if scale != 1.:
                new_w = int(w*scale)
                new_h = int(h*scale)
                img2 = img2.resize((new_w,new_h),Image.BILINEAR)
                target2 = target2.resize((new_w,new_h),Image.NEAREST)
        img2 = np.float32(np.array(img2)) / 255.
        target2 = np.array(target2)
        img2, target2 = self.rand_crop(img2,target2)

        img2 = self.img_transform(img2)
        target2 = self.segm_transform(target2)
        ###########

        
        
        return img, segm,img2,target2




#############
class MultiScaleTrainDataset(torch.utils.data.Dataset):
    def __init__(self, imgroot, gtroot):
        # parse options
        self.scale = [1.,1.5,2.]
        self.cropsize = (1080,1920)
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        imglist_= os.listdir(imgroot)
        self.imglist=[]
        for x in imglist_:
            if x.split('.')[-1]=='jpg':            
                self.imglist.append(x)
        
        self.imgroot = imgroot
        self.gtroot = gtroot

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        if h<=1080 or w<=1920:
            return image,label
#        image = self.pad_image(image, h, w, self.crop_size,
#                                (0.0, 0.0, 0.0))
#        label = self.pad_image(label, h, w, self.crop_size,
#                                (self.ignore_label,))

#        new_h, new_w = label.shape
        x = random.randint(0, w  - self.cropsize[1])
        y = random.randint(0, h - self.cropsize[0])
        image = image[y:y+self.cropsize[0], x:x+self.cropsize[1]]
        label = label[y:y+self.cropsize[0], x:x+self.cropsize[1]]

        return image, label
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(segm).float()
        segm = segm.unsqueeze(0)
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self,idx):
        imgname = self.imglist[idx]
        img_path = os.path.join(self.imgroot,imgname)
        gtname = imgname+'.png'
        gt_path = os.path.join(self.gtroot,gtname)
        img = Image.open(img_path).convert('RGB')
        segm = Image.open(gt_path)
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

            # random_flip
        if np.random.choice([0, 1]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        scale = np.random.choice(self.scale)
        if scale != 1.:
            w,h = img.size
            new_w = int(w*scale)
            new_h = int(h*scale)
            img = img.resize((new_w,new_h),Image.BILINEAR)
            segm = segm.resize((new_w,new_h),Image.NEAREST)
        img = np.float32(np.array(img)) / 255.
        segm = np.array(segm)
        img, segm = self.rand_crop(img,segm)
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
        
        return img, segm



class TrainDataset(BaseDataset):
    def __init__(self, imgroot,gtroot, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(imgroot,gtroot, opt, **kwargs)
        #self.root_dataset = root_dataset
        # down sampling rate of segm labe
        #self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        #self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        #self.cur_idx = 0
        #self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample



DATAROOT="/your/path/to/LVSP_plus_data_label124_480p"


#####
#ARCH=res101_ocrnet
#CFG="config/vsp-resnet101dilated-ocr_deepsup.yaml"
####

#ARCH=res101_deeplab
#CFG="config/vsp-resnet101dilated-deeplab.yaml"
#CFG="config/vsp-resnet50dilated-deeplab.yaml"

######
#ARCH=res101_nonlocal2d_nodown
#CFG="config/vsp-resnet101dilated-nonlocal2d.yaml"
########
#ARCH=mobile_ppm
ARCH=res101_ppm
CFG="config/vsp-resnet101dilated-ppm_deepsup.yaml"
#CFG="config/ade20k-resnet50dilated-ppm_deepsup.yaml"
#CFG="config/ade20k-mobilenetv2dilated-ppm_deepsup.yaml"

#ARCH=resnet101uper
#CFG="config/ade20k-resnet101-upernet.yaml"
#CFG="config/ade20k-resnet50-upernet.yaml"


#PREDIR="../data/imgnetpre/resnet101-imagenet.pth"
PREDIR='./imgnetpre/resnet101-imagenet.pth'
#PREDIR="../data/imgnetpre/resnet50-imagenet.pth"
#PREDIR="../data/imgnetpre/mobilenet_v2.pth.tar"


#SAVE="../afs/video_seg/vsp_124"
SAVE="./savemodel"

#ARCH='hrnet'
#CFG="config/ade20k-hrnetv2.yaml"
#PREDIR="../data/imgnetpre/hrnetv2_w48-imagenet.pth"

DATAROOT2=../data/adeour
BATCHSIZE=8
WORKERS=12
USETWODATA=False
START_GPU=0
GPU_NUM=2
TRAINFPS=2
LR=0.002



CROPSIZE=479

LESSLABEL=False


USE_CLIPDATASET=True
EPOCH=120
NAME='job_lr'$LR'batchsize'$BATCHSIZE'_EPOCH'$EPOCH'_FPS'$TRAINFPS"_arch"$ARCH"new124_gpu"$GPU_NUM"_480p""USE_CLIPDATASET"$USE_CLIPDATASET
SAVEROOT=$SAVE"/"$NAME
VAL=False
echo $CFG


echo 'train...'
python train.py --cfg $CFG  --predir $PREDIR --batchsize $BATCHSIZE --workers $WORKERS --start_gpu $START_GPU --gpu_num $GPU_NUM --dataroot $DATAROOT --trainfps $TRAINFPS --lr $LR --multi_scale True --saveroot $SAVEROOT --totalepoch $EPOCH --dataroot2 $DATAROOT2 --usetwodata $USETWODATA --cropsize $CROPSIZE --validation $VAL --lesslabel $LESSLABEL --use_clipdataset $USE_CLIPDATASET


LOAD_EN=$SAVEROOT'/encoder_epoch_'$EPOCH'.pth'
LOAD_DE=$SAVEROOT'/decoder_epoch_'$EPOCH'.pth'



TESTBATCHSIZE=2
ISSAVE=True
IMGSAVEROOT='./saveimg/'$NAME'_train'
USE720p=False
LESSLABLE=False

echo 'val...'
python test.py --cfg $CFG   --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT --load_en $LOAD_EN --load_de $LOAD_DE --batchsize $TESTBATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --split 'val'
echo 'test...'

python test.py --cfg $CFG    --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT --load_en $LOAD_EN --load_de $LOAD_DE --batchsize $TESTBATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --split 'test'


DATAROOT="/your/path/to/LVSP_plus_data_label124_480p"


#####
ARCH=res101_ocrnet
CFG="config/vsp-resnet101dilated-ocr_deepsup.yaml"
####



PREDIR='./imgnetpre/resnet101-imagenet.pth'


SAVE="./savemodel"


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
ISSAVE=False
IMGSAVEROOT='./saveimg/'$NAME'_train'
USE720p=False
LESSLABLE=False

echo 'val...'
python test.py --cfg $CFG   --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT --load_en $LOAD_EN --load_de $LOAD_DE --batchsize $TESTBATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --split 'val'
echo 'test...'

python test.py --cfg $CFG    --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT --load_en $LOAD_EN --load_de $LOAD_DE --batchsize $TESTBATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --split 'test'


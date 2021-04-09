DATAROOT="your/path/to/VSPW_480p"



SAVE="./savemodel"
DATAROOT2='data2'
BATCHSIZE=8
WORKERS=12
CROPSIZE=479



START_GPU=0
GPU_NUM=4
TRAINFPS=1
EPOCH=120
LR=0.002
VAL=False
USETWODATA=False
LESSLABEL=False
CLIPNUM=4
DILATION=0
CLIPUP=False
CLIPMIDDLE=False
OTHERGT=False
PROPCLIP2=False
EARLYFUSE=True
EARLYCAT=False
CONVLSTM=False
NON_LOCAL=False



FIX=False
ALLSUP=True
ALLSUPSCALE=0.5
LINEAR_COM=True
DISTSOFTMAX=False
DISTNEAREST=False
TEMP=0.05

DILATION2="3,6,9"

CLIPOCR_ALL=False
USEMEMORY=True

METHOD='clip_psp'



PREROOT=''
PRE_ENC="./imgnetpre/resnet101-imagenet.pth"
MAXDIST='3'
#########
ARCH=resnet101
CFG='vsp-'$ARCH'dilated-ppm_deepsup_clip.yaml'
#CFG='vsp-'$ARCH'dilated-ppm_clip.yaml'
#CFG="vsp-"$ARCH"dilated_tdnet.yaml"
PREDIR="../data/imgnetpre/"$ARCH"-imagenet.pth"

NAME='newjob_lr'$LR'_bs'$BATCHSIZE'_epoch'$EPOCH'_FPS'$TRAINFPS'_clipnum'$CLIPNUM"_dilation"$DILATION"_fix"$FIX"_tdnet"$TDNET"_arch"$ARCH'_method'$METHOD'_DISTSOFTMAX'$DISTSOFTMAX'_DISTNEAREST'$DISTNEAREST"_CLIPOCR_ALL"$CLIPOCR_ALL"_USEMEMORY"$USEMEMORY"imgnetpre"


SAVEROOT=$SAVE"/"$NAME
python train_clip2.py --cfg config/$CFG  --predir $PREDIR --batchsize $BATCHSIZE --workers $WORKERS --start_gpu $START_GPU --gpu_num $GPU_NUM --dataroot $DATAROOT --trainfps $TRAINFPS --lr $LR --multi_scale True --saveroot $SAVEROOT --totalepoch $EPOCH  --dataroot2 $DATAROOT2 --usetwodata $USETWODATA --cropsize $CROPSIZE --validation $VAL --lesslabel $LESSLABEL --clip_num $CLIPNUM --dilation_num $DILATION --clip_up $CLIPUP --clip_middle $CLIPMIDDLE --fix $FIX --othergt $OTHERGT --propclip2 $PROPCLIP2  --earlyfuse $EARLYFUSE      --early_usecat $EARLYCAT   --allsup $ALLSUP --allsup_scale $ALLSUPSCALE --linear_combine $LINEAR_COM --distsoftmax $DISTSOFTMAX --distnearest $DISTNEAREST --temp $TEMP --pre_enc $PRE_ENC --max_distances $MAXDIST   --method $METHOD --dilation2 $DILATION2 --clipocr_all $CLIPOCR_ALL --use_memory $USEMEMORY



###inference
echo 'val'
BATCHSIZE=1
GPU_NUM=1
ISSAVE=True
LESSLABLE=False
USE720p=False
EARLYFUSE=False
EARLYCAT=False

#CLIPNUM=5

IMGSAVEROOT='./clipsaveimg/'$NAME

LOAD=$SAVEROOT'/model_epoch_'$EPOCH'.pth'

python test_clip2.py --cfg config/$CFG --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT  --batchsize $BATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --clip_num $CLIPNUM --dilation_num $DILATION --load $LOAD --split 'val'  --allsup $ALLSUP --allsup_scale $ALLSUPSCALE --linear_combine $LINEAR_COM --distsoftmax $DISTSOFTMAX --distnearest $DISTNEAREST --temp $TEMP --max_distances $MAXDIST --gpu_num $GPU_NUM  --method $METHOD --dilation2 $DILATION2 --clipocr_all $CLIPOCR_ALL --use_memory $USEMEMORY

echo 'test'

python test_clip2.py --cfg config/$CFG --start_gpu $START_GPU --dataroot $DATAROOT --saveroot $IMGSAVEROOT  --batchsize $BATCHSIZE --is_save $ISSAVE --lesslabel $LESSLABLE --use_720p $USE720p --clip_num $CLIPNUM --dilation_num $DILATION --load $LOAD --split 'test'   --allsup $ALLSUP --allsup_scale $ALLSUPSCALE --linear_combine $LINEAR_COM --distsoftmax $DISTSOFTMAX --distnearest $DISTNEAREST --temp $TEMP --max_distances $MAXDIST --gpu_num $GPU_NUM --method $METHOD --dilation2 $DILATION2 --clipocr_all $CLIPOCR_ALL --use_memory $USEMEMORY





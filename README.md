# VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild

A pytorch implementation of the CVPR2021 paper "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild"

# Preparation

## Dependencies
 - Python 3.7
 - Pytorch 1.5
 - Numpy

Download the ImageNet-pretrained models at [this link](https://drive.google.com/file/d/1VFmObwlx4d_K7FOjFNk5LhEb3jP8_NaD/view?usp=sharing). Put it in the root folder and decompress it.

# Train and Test

Edit the *.sh* files in *scripts/* and change the **$DATAROOT** to your path to VSPW. 

## Image-based methods

PSPNet

```
sh scripts/run_psp.sh
```

OCRNet

```
sh scripts/run_ocr.sh
```

## Video-based methods

TCB-PSP

```
sh run_temporal_psp.sh
```

TCB-OCR

```
sh run_temporal_ocr.sh
```



This implementation utilized [this code](https://github.com/CSAILVision/semantic-segmentation-pytorch) and [RAFT](https://github.com/princeton-vl/RAFT).



# Citation

```
@inproceedings{miao2021vspw,

  title={VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild},

  author={Miao, Jiaxu and Wei, Yunchao and  Wu, Yu and Liang, Chen and Li, Guangrui and Yang, Yi},

  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},

  year={2021}

}
```



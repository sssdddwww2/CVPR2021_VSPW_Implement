# VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild

A pytorch implementation of the CVPR2021 paper "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild"

# Preparation

## Download VSPW dataset

The VSPW dataset with extracted frames and masks is available [here](https://github.com/sssdddwww2/vspw_dataset_download). Now you can directly download [VSPW_480P dataset](https://github.com/sssdddwww2/vspw_dataset_download).

## Dependencies
 - Python 3.7
 - Pytorch 1.3.1
 - Numpy

Download the ImageNet-pretrained models at [this link](https://drive.google.com/file/d/1VFmObwlx4d_K7FOjFNk5LhEb3jP8_NaD/view?usp=sharing). Put it in the root folder and decompress it.

# Train and Test

Resize the frames and masks of the VSPW dataset to *480p*. 

```
python change2_480p.py
```

Edit the *.sh* files in *scripts/* and change the **$DATAROOT** to your path to VSPW_480p. 

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

## Evaluation on TC and VC

Change dataroot and prediction root in *TC_cal.py* and *VC_perclip.py*.

```
python TC_cal.py
```

```
python VC_perclip.py
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



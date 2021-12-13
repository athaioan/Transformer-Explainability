# Learning Pixel-level Semantic Affinity with Image-level Supervision

**Original repo: https://github.com/jiwoon-ahn/irn instead.**

## Important resources

### Weights:
Hybrid Pascal Voc Weights: https://drive.google.com/drive/folders/1bPtIrgnwQaOhCjBuQjL4S1lMBQDBmEjE?usp=sharing

### CRFs:
* Low: https://1drv.ms/u/s!ArK_bULPXnoqhYEOXFAks9NZsiJPCw?e=FKANlR
* High: https://1drv.ms/u/s!ArK_bULPXnoqhYENdfk_sy3FNgfojA?e=aLdV2e

## Introduction

The replicated code of:

Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation, Jiwoon Ahn and Suha Kwak, CVPR 2018 [[Paper]](https://arxiv.org/abs/1803.10464)

We have developed a framework based on AffinityNet to generate accurate segmentation labels of training images given their image-level class labels only. A segmentation network learned with our synthesized labels outperforms previous state-of-the-arts by large margins on the PASCAL VOC 2012.

## Reference
```
@InProceedings{Ahn_2018_CVPR,
author = {Ahn, Jiwoon and Kwak, Suha},
title = {Learning Pixel-Level Semantic Affinity With Image-Level Supervision for Weakly Supervised Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## Prerequisite
* Tested on Windows 10, with Python 3.8, PyTorch 1.9 and CUDA 11.4 (Pascal).

## Results and Trained Models
#### Class Activation Map

| Model         | Train (mIoU)    | Val (mIoU)    | |
| ------------- |:-------------:|:-----:|:-----:|
| VGG-16        | 48.9 | 46.6 | [[Weights]](https://drive.google.com/file/d/1Dh5EniRN7FSVaYxSmcwvPq_6AIg-P8EH/view?usp=sharing) |
| ResNet-38     | 47.7 | 47.2 | [[Weights]](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing) |
| ResNet-38     | 48.0 | 46.8 | CVPR submission |

#### Random Walk with AffinityNet

| Model         | alpha | Train (mIoU)    | Val (mIoU)    | |
| ------------- |:-----:|:---------------:|:-------------:|:-----:|
| VGG-16        | 4/16/32 | 59.6 | 54.0 | [[Weights]](https://drive.google.com/file/d/10ue1B20Q51aQ53T93RiaiKETlklzo4jp/view?usp=sharing) |
| ResNet-38     | 4/16/32 | 61.0 | 60.2 | [[Weights]](https://drive.google.com/open?id=1mFvTH3siw0SS0vqPH0o9N3cI_ISQacwt) |
| ResNet-38     | 4/16/24 | 58.1 | 57.0 | CVPR submission |

>*beta=8, gamma=5, t=256 for all settings

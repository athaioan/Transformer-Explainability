# Reproduction of Transformer Interpretability Beyond Attention Visualization [CVPR 2021]

## Requirements

### Create and activate a virtual environment

Creation and activation of virtual environments in Linux systems is done by executing the command venv:
```bash
python3 -m venv /path/to/new/virtual/environment
. /path/to/new/virtual/environment/bin/activate
```

When using Anaconda Creation and activation of virtual environments in Linux systems is done by executing the following command:
```bash
conda create --name /path/to/new/virtual/environment python=3.8 numpy
source activate /path/to/new/virtual/environment
```

### Install dependencies
Installation of the required libraries to run our code can be achieved by the following command:
```bash
pip install -r requirements.txt
```

### Download the different datasets
Downloading ImageNet 2012 validation set, ImageNet segmentation dataset and PascalVOC 2012, can be achieved by running the following commands respectively:
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
```

## Experiments
### ImageNet
This can be used to run our demo:
```bash
python3 demo_eval/demo.py
```


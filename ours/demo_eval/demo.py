import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace
from utils import *
import torchvision.transforms as transforms

### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=226,
                       input_dim_crop=224,
                       pretrained_weights="pretrained/vgg16_20M.pth",
                       val_set="ILSVRC2012_img_val",
                       test_set="test_new.txt",
                       labels_dict="val_labels_dict.npy")


import pickle

dict = {}

# with open('C:/Users/johny/Desktop/Transformer-Explainability-main/Transformer-Explainability-main/val_labels.txt') as file:
#     lines = file.readlines()
#     lines = [line.rstrip() for line in lines]
#     for current_line in lines:
#
#         key, value = current_line.split(" ")
#         value = int(value)
#         dict[key] = value
#
# with open("val_labels_dict.npy", 'wb') as f:
#     pickle.dump(dict, f)

# loading
# with open(args.labels_dict, 'rb') as f:
#     ret_di = pickle.load(f)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(args.input_dim),
    transforms.CenterCrop(args.input_dim_crop),
    transforms.ToTensor(),
    normalize,
])



## Constructing the training loader
val_loader = ImageNetVal(args.val_set, args.labels_dict, transform) ## loading val split (50.000)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

for data in val_loader:

    print("")
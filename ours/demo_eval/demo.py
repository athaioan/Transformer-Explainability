import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace

from utils import *
from network import ViT_model as model


### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=224,
                       pretrained_weights="pretrained/vgg16_20M.pth",
                       val_set="ILSVRC2012_img_val",
                       labels_dict="val_labels_dict.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                       )



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
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
    normalize,
])

# transform_gt_mask = transforms.Compose([
#     transforms.Resize(args.input_dim),
#     transforms.ToTensor(),
# ])


## Constructing the training loader
val_loader = ImageNetVal(args.val_set, args.labels_dict, args.device, transform) ## loading val split (50.000)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

for index, data in enumerate(val_loader):

    ## TODO model
    img = data[0]
    label = data[1]

    model.extract_cam(img,label)

    print("")
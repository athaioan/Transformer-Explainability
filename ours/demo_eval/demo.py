import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
# from utils import *
# from network import ViT_model
import pickle

from ours.Utils.utils import * # Georgios
from ours.Networks.network import ViT_model # Georgios

### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=224,
                       pretrained_weights="saved_weights.pth",
                       # val_set="ILSVRC2012_img_val",
                       val_set="C:/Users/johny/Desktop/Transformer-Explainability-main/Transformer-Explainability-main\Dataset_1/val/n01440764/",
                       val_set_semg="gtsegs_ijcv.mat",
                       labels_dict="val_labels_dict.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                       )

# args.val_set_semg = r'C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\gtsegs_ijcv.mat'

dict = {}

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform_perturb = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
    normalize,
])

transform_gt_mask = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
])


# # # ## Constructing the validation loader
val_loader = ImageNetVal(args.val_set, args.labels_dict, args.device, transform_perturb) ## loading val split (50.000)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

# Constructing the validation segm loader
# val_loader = ImageNetSegm(args.val_set_semg, args.device, transform, transform_gt_mask) ## loading val split (50.000)
# val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)


## Initialize model
model = ViT_model(device=args.device) ## TODO inster the number of class imagenet:1000 , PascalVOC: 18
model.load_pretrained(args.pretrained_weights)
model.eval()
model.zero_grad()

# (n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_dim=768,
#                  n_heads=12, QKV_bias=False, att_dropout=0., out_dropout=0., n_block=12, mlp_ratio=4.)

# for index, data in enumerate(val_loader):
#
#     print(index/len(val_loader))
#
#     ## TODO model
#     img = data[0]
#     label = data[1]
#     # img_orig = data[2]
#
#     # preds = model(img)
#
#     explainability_cue, preds = model.extract_LRP(img)
#
#     eval_batch(explainability_cue, labels)
#
#     # correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, model, batch_idx)
#
#
#     print("")


# pixAcc, mIoU, mAp = model.extract_metrics(val_loader)
# print(pixAcc, mIoU, mAp)
# 0.7971615078627627 0.6198494989665153 0.8604044712192599


AUC = model.extract_AUC(val_loader, normalize, positive=True, vis_class_top=True)
print(AUC)

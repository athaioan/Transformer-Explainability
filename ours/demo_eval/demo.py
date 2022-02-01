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
                       val_set=r"C:\Users\georg\PycharmProjects\Transformer-Explainability\ours\ILSVRC2012_img_val",
                       # val_set="C:/Users/johny/Desktop/Transformer-Explainability-main/ours/ILSVRC2012_img_val",
                       val_set_semg="gtsegs_ijcv.mat",
                       labels_dict="val_labels_dict.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                       )

args.val_set_semg = r'C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\gtsegs_ijcv.mat'

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
val_loader_segm = ImageNetSegm(args.val_set_semg, args.device, transform, transform_gt_mask) ## loading val semgntation split (4.276)
val_loader_segm = DataLoader(val_loader_segm, batch_size=args.batch_size, shuffle=False)


## Initialize model
model = ViT_model(device=args.device)
model.load_pretrained(args.pretrained_weights)
model.eval()
model.zero_grad()


# pixAcc, mIoU, mAp = model.extract_metrics(val_loader_segm)
# print(pixAcc, mIoU, mAp)
# # 0.7972802107022943 0.6198160110748642 0.8603164970095842


AUC = model.extract_AUC(val_loader, normalize, positive=True, vis_class_top=True)
print(AUC)
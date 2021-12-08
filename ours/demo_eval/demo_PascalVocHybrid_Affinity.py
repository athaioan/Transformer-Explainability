import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_hybrid_model_Affinity
import pickle

# from ours.Utils.utils import * # Georgios
# from ours.Networks.network import ViT_model # Georgios


### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=448,
                       # pretrained_weights="saved_weights_VOC.pth",
                       pretrained_weights="PascalVOC_classification_Hybrid_1/stage_1.pth",
                       epochs=15,
                       lr=5e-3,
                       weight_decay=1e-4,
                       VocClassList="C:/Users/johny/Desktop/Transformer-Explainability-main/ours/PascalVocClasses.txt",
                       voc12_img_folder="VOCdevkit/VOC2012/JPEGImages/",
                       train_set=r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt", ## TODO train set
                       # val_set=r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt",
                       low_cams_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/ours/val_cams/crf_lows/",
                       high_cams_fold = "C:/Users/johny/Desktop/Transformer-Explainability-main/ours/val_cams/crf_highs/",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       )


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_loader = PascalVOC2012Affinity(args.train_set,  args.voc12_img_folder, args.low_cams_fold, args.high_cams_fold,
                              args.input_dim, args.device,

                              img_transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                              normalize,
                              ]),

                             label_transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 ResizeMultipleChannels((args.input_dim, args.input_dim), mode='bilinear'),
                             ]),
                             both_transform=transforms.RandomHorizontalFlip(p=0.5))

train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=False)


model = ViT_hybrid_model_Affinity(max_epochs=args.epochs, device=args.device)


model.load_pretrained(args.pretrained_weights)
model.session_name = "PascalVOC_classification_Hybrid_Affinity_1"
model.eval()


for data in train_loader:

    img = data[1]

    model(img)

    print("")






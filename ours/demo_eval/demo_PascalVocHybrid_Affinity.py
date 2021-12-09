import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
# from utils import *
# from network import ViT_hybrid_model_Affinity
import pickle

from ours.Utils.utils import * # Georgios
from ours.Networks.network import * # Georgios


### Setting arguments
args = SimpleNamespace(batch_size=1,
                       input_dim=448,
                       # pretrained_weights="saved_weights_VOC.pth",
                       pretrained_weights="PascalVOC_classification_Hybrid_1/stage_1.pth",
                       epochs=15,
                       lr=5e-3,
                       weight_decay=1e-4,
                       VocClassList=r"C:\Users\georg\PycharmProjects\Transformer-Explainability\ours\txts\pvclasses.txt",
                       voc12_img_folder="C:/Users/georg/Documents/KTH_ML_Master/Deep Learning Advanced Course/Project/Datasets/VOCdevkit/VOC2012/JPEGImages",
                       train_set=r"C:\Users\georg\PycharmProjects\Transformer-Explainability\ours\txts\val.txt", ## TODO train set
                       # val_set=r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt",
                       low_cams_fold = r"C:\Users\georg\PycharmProjects\Transformer-Explainability\ours\PascalVOC_classification_Hybrid_1\val_cams\crf_highs",
                       high_cams_fold = r"C:\Users\georg\PycharmProjects\Transformer-Explainability\ours\PascalVOC_classification_Hybrid_1\val_cams\crf_highs",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       tol=1e-5
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

for iterator in train_loader:

    _, im_orig, gt_mask, im_dims = iterator # unpack

    affinities = model(im_orig)

    labels = [gt_mask[i].to(args.device, non_blocking=True) for i in range(3)]
    counts = [torch.sum(labels[i]) + args.tol for i in range(3)]
    losses = [affinity_ce_losses(labels[i], affinities, counts[i], i) for i in range(3)]

    model_loss = 0.5 * (losses[0]/2 + losses[1]/2 + losses[2])

    model.optimizer.zero_grad()
    model_loss.backward()
    model.optimizer.step()

torch.save(model.module.state_dict(), model.session_name + '_mytry.pth')






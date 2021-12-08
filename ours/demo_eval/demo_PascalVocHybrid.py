import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_hybrid_model
import pickle

# from ours.Utils.utils import * # Georgios
# from ours.Networks.network import ViT_model # Georgios

### Setting arguments
args = SimpleNamespace(batch_size=3,
                       input_dim=448,
                       pretrained_weights="Resnet_ViT.pth",
                       epochs=20,
                       lr=5e-3,
                       weight_decay=1e-4,
                       VocClassList="C:/Users/johny/Desktop/Transformer-Explainability-main/ours/PascalVocClasses.txt",
                       voc12_img_folder="VOCdevkit/VOC2012/JPEGImages/",
                       train_set=r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012\ImageSets\Segmentation\train_augm.txt",
                       val_set=r"C:\Users\johny\Desktop\Transformer-Explainability-main\ours\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt",
                       labels_dict="VOCdevkit/VOC2012/cls_labels.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       )


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_loader = PascalVOC2012(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                              normalize,
                              ]))

train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)

val_loader = PascalVOC2012(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.Resize((args.input_dim, args.input_dim)),
                              transforms.ToTensor(),
                              normalize,
                              ]))
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

## Initialize model
model = ViT_hybrid_model(img_size=(448, 448), patch_size=16, n_heads=12, n_blocks=12,  embed_size=768, n_classes=20,
                         max_epochs=args.epochs, device=args.device) ## TODO inster the number of class imagenet:1000 , PascalVOC: 18


model.load_pretrained(args.pretrained_weights)
model.session_name = "PascalVOC_classification_Hybrid_1"
model.eval()

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)


# Prepare optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay)


for index in range(model.max_epochs):

    for g in optimizer.param_groups:
        g['lr'] = args.lr * (1-index/model.max_epochs)

    print("Training epoch...")
    model.train_epoch(train_loader, optimizer)

    print("Validating epoch...")
    model.val_epoch(val_loader)

    model.visualize_graph()

    if model.val_history["loss"][-1] < model.min_val:
        print("Saving model...")
        model.min_val = model.val_history["loss"][-1]

        torch.save(model.state_dict(), model.session_name+"/stage_1.pth")


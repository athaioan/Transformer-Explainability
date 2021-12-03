import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from types import SimpleNamespace
from utils import *
from network import ViT_model
import pickle

# from ours.Utils.utils import * # Georgios
# from ours.Networks.network import ViT_model # Georgios

### Setting arguments
args = SimpleNamespace(batch_size=16,
                       input_dim=448,
                       pretrained_weights="saved_weights.pth",
                       epochs=20,
                       lr=0.04,
                       weight_decay=5e-4,
                       voc12_img_folder="C:/Users/georg/Documents/KTH_ML_Master/Deep Learning Advanced Course/Project/Datasets/VOCdevkit/VOC2012/JPEGImages/",
                       train_set=r"C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\VOCdevkit\VOC2012\ImageSets\Segmentation2\train_augm.txt",
                       val_set=r"C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\VOCdevkit\VOC2012\ImageSets\Segmentation2\val.txt",
                       labels_dict=r"C:\Users\georg\Documents\KTH_ML_Master\Deep Learning Advanced Course\Project\Datasets\VOCdevkit\VOC2012\cls_labels.npy",
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       step_update_lr=4,
                       )

# lines= [ ]
# with open(r"C:\Users\johny\Desktop\KTH COURSES\Deep Learning\Group90_Project\VOCdevkit\VOC2012\ImageSets\Segmentation\train_aug.txt") as file:
#     for line in file:
#         line = line.strip() #or some other preprocessing
#         lines.append(line) #storing everything in memory!

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_loader = PascalVOC2012(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                              normalize,
                              ]))
train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)

val_loader = PascalVOC2012(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim, args.device,
                              transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                              ]))
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)


## Initialize model
model = ViT_model(img_size=(448, 448), patch_size=32, n_classes=20, max_epochs=args.epochs, device=args.device) ## TODO inster the number of class imagenet:1000 , PascalVOC: 18
model.load_pretrained("saved_weights.pth")
model.session_name = "PascalVOC_classification"
model.eval()

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)

# change lr and weight decay
# args.lr = 0.03
# args.weight_decay = 0.1
args.momentum = 0.9

# Prepare optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

optimizer2 = torch.optim.Adam(model.parameters(),
                            lr=0.04)

for index in range(model.max_epochs):

    if model.current_epoch % args.step_update_lr == args.step_update_lr - 1:
        # for g in optimizer.param_groups:
            # g['lr'] = g['lr'] / 3

        for g in optimizer2.param_groups:
            g['lr'] = g['lr'] / 3


    print("Training epoch...")
    # model.train_epoch(train_loader, optimizer)
    model.train_epoch(train_loader, optimizer2)

    print("Validating epoch...")
    model.val_epoch(val_loader)

    model.visualize_graph()

    if model.val_history["loss"][-1] < model.min_val:
        print("Saving model...")
        model.min_val = model.val_history["loss"][-1]

        torch.save(model.state_dict(), model.session_name+"/stage_1.pth")

#
# ## Constructing the validation loader
# val_loader = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
# val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False) ## no point in shufflying the validation data
#

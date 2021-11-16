from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt


class ImageNetVal(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_folder, labels_dict, transform):

        self.labels_dict = np.load(labels_dict, allow_pickle=True)
        self.transform = transform
        self.img_names = os.listdir(img_folder)
        self.img_names = np.asarray([img_folder+"/"+current_img for current_img in self.img_names])


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        current_path = self.img_names[idx]

        img_orig = Image.open(current_path)

        img = self.transform(img_orig)



        return img_meta, img, label
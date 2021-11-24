import torch
from torch import nn
import matplotlib.pyplot as plt
from overwritten_layers import *
from ours.Utils.utils import *

class ViT_model(nn.Module):
    def __init__(self, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_dim=768):
        super(ViT_model, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_dim = embed_dim

        self.patching = Img_to_Patch(img_size, patch_size, in_ch, embed_dim)

        self.positional_embed = nn.Parameter(torch.zeros(1, self.patching.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        no_grad_trunc_normal_(self.positional_embed, std=.02)
        no_grad_trunc_normal_(self.cls_token, std=.02)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patching(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embed

        print("")

    def extract_cam(self):
        print("")

    def load_pretrained(self):
        print("")


class Img_to_Patch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, input_ch=3, embed_size=768):
        super(Img_to_Patch, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_ch = input_ch
        self.embed_size = embed_size
        ## TODO architecture
        self.conv2d = Conv2d(self.input_ch, self.embed_size, kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size))

        self.n_patches = (img_size[1] // self.patch_size) * (img_size[0] // self.patch_size)

    def forward(self, x):

        x = self.conv2d(x).flatten(2).transpose(1, 2)
        return x
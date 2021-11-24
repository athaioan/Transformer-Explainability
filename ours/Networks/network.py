import torch
from torch import nn
import matplotlib.pyplot as plt
from overwritten_layers import *
from utils import *
from einops import rearrange


class ViT_model(nn.Module):
    def __init__(self, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_dim=768):
        super(ViT_model, self).__init__()


        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_dim = embed_dim

        self.add = Add()
        self.patching = Img_to_patch(img_size, patch_size, in_ch, embed_dim)

        self.positional_embed = nn.Parameter(torch.zeros(1, self.patching.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        set_seeds(0)
        no_grad_trunc_normal_(self.positional_embed, std=.02)
        no_grad_trunc_normal_(self.cls_token, std=.02)

        self.input_grad = None
        self.att = Attention_layer(768, n_heads=12, QKV_bias=True, att_dropout=0., proj_dropout=0.)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patching(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = self.add([x, self.positional_embed])# x+= self.positional_embed

        x.register_hook(self.store_input_grad) ## When computing the grad wrt to the input x, store that grad to the model.input_grad

        self.att(x)

        print("")

    def store_input_grad(self, grad):
        self.input_grad = grad


    def extract_cam(self):
        print("")

    def load_pretrained(self):
        print("")


class Img_to_patch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, input_ch=3, embed_size=768):
        super(Img_to_patch, self).__init__()
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

class Attention_layer(nn.Module):

    def __init__(self, embed_size, n_heads=12, QKV_bias=False, att_dropout=0., proj_dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        head_dim = embed_size // n_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = Linear(embed_size, embed_size * 3, bias=QKV_bias)

        # A = Q*K^T
        self.matmul1 = Matmul(transpose=True)
        self.matmul2 = Matmul(transpose=False)

        self.softmax = Softmax()


        # self.matmul1 = matmul()
        # # attn = A*V
        # self.matmul2 = einsum('bhij,bhjd->bhid')

        # self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        # self.att_drop = Dropout(att_dropout)
        # self.proj = Linear(dim, dim)
        # self.proj_drop = Dropout(proj_drop)
        # self.softmax = Softmax(dim=-1)
        #
        # self.attn_cam = None
        # self.attn = None
        # self.v = None
        # self.v_cam = None
        # self.attn_gradients = None

    def store_v(self, v):
        self.v = v

    def forward(self,x):
        batch, n, embed_size = x.shape
        qkv = self.qkv(x)

        ## ours
        qkv = torch.reshape(qkv, (batch, n, 3, self.n_heads, embed_size//self.n_heads))
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        self.store_v(v)

        scaled_products = self.matmul1([q, k]) * self.scale

        attn = self.softmax(scaled_products)

        print("")
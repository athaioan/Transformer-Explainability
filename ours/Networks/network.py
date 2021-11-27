import torch
from torch import nn
import matplotlib.pyplot as plt
from overwritten_layers import *
from utils import *
from einops import rearrange
# from ours.Utils.utils import *


class ViT_model(nn.Module):
    def __init__(self, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_size=768,
                 n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0., n_blocks=12, mlp_hidden_ratio=4.,
                 device="cuda"):
        super(ViT_model, self).__init__()

        self.device = device
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.QKV_bias = QKV_bias

        self.add = Add()
        self.patch_embed = Img_to_patch(img_size, patch_size, in_ch, embed_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        set_seeds(0)
        no_grad_trunc_normal_(self.pos_embed, std=.02)
        no_grad_trunc_normal_(self.cls_token, std=.02)

        self.input_grad = None

        self.blocks = nn.ModuleList([Block(embed_size=self.embed_size, n_heads=self.n_heads,
                           QKV_bias=self.QKV_bias, att_dropout=att_dropout, out_dropout=out_dropout, mlp_hidden_ratio=4)
                                       for _ in range(self.n_blocks)])

        self.norm = LayerNorm(embed_size)
        self.head = Linear(self.embed_size, self.n_classes)
        self.pool = ClsSelect()

        self.to(self.device)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = self.add([x, self.pos_embed])# x+= self.positional_embed

        x.register_hook(self.store_input_grad) ## When computing the grad wrt to the input x, store that grad to the model.input_grad

        for current_block in self.blocks:
            x = current_block(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, index=torch.tensor(0, device=x.device)) ## retrieve the cls
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def store_input_grad(self, grad):
        self.input_grad = grad


    def relevance_progation(self, one_hot_label, start_layer=0):

        ## from top to bottom
        relevance = self.head.relevance_propagation(one_hot_label)
        relevance = self.pool.relevance_propagation(relevance)
        relevance = self.norm.relevance_propagation(relevance)

        for current_block in reversed(self.blocks):
            # relevance = current_block.relevance_propagation(relevance)

            # ## for purposes of evaluating the attention rel_prop without having block's
            # ## rel_pro beforehand
            # relevance = torch.load("tensor.pt").to(self.device) ## loading the
            # ## input relevance at attention as generated from her implementation
            # relevance = current_block.attn.relevance_propagation(relevance)
            # print("")

        return relevance


    def extract_LRP(self, input, class_indices = None, start_layer=0):

        pred = self(input)

        if class_indices is None:
            class_indices = torch.argmax(pred, dim=1).data.cpu().numpy().tolist()


        one_hot = np.zeros((1, pred.shape[-1]), dtype=np.float32)
        one_hot[0, class_indices] = 1

        one_hot_label = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input.device) * pred)

        self.zero_grad()
        one_hot.backward(retain_graph=True) ## Register_hooks are excecuted in here

        cam = self.relevance_progation(torch.tensor(one_hot_label).to(input.device),
                                 start_layer=start_layer)

        return cam



    def load_pretrained(self, weights_path):

        ## loading weights
        weights_dict = torch.load(weights_path)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}

        no_pretrained_dict = {k: v for k, v in model_dict.items() if
                           not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)




class Img_to_patch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, input_ch=3, embed_size=768):
        super(Img_to_patch, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_ch = input_ch
        self.embed_size = embed_size
        ## TODO architecture
        self.proj = Conv2d(self.input_ch, self.embed_size, kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size))

        self.n_patches = (img_size[1] // self.patch_size) * (img_size[0] // self.patch_size)

    def forward(self, x):

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention_layer(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=False, att_dropout=0., out_dropout=0.):
        super().__init__()

        self.n_heads = n_heads
        self.QKV_bias = QKV_bias


        head_dim = embed_size // n_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = Linear(embed_size, embed_size * 3, bias=self.QKV_bias)
        self.proj = Linear(embed_size, embed_size)

        # A = Q*K^T
        self.matmul1 = Matmul(transpose=True)
        # att = A*V
        self.matmul2 = Matmul(transpose=False)
        self.att_softmax = Softmax(dim=-1)


        self.att_dropout = Dropout(att_dropout)
        self.out_dropout = Dropout(out_dropout)

        self.v = None
        self.att = None
        self.att_grad = None

        self.v_relevance = None
        self.att_relevance = None

    # TODO eliminated those not being used

    def store_v(self, v):
        self.v = v

    def store_att(self, att):
        self.att = att

    def store_att_grad(self, grad):
        self.att_grad = grad

    def store_v_relevance(self, relevance):
        self.v_relevance = relevance

    def store_att_relevance(self, relevance):
        self.att_relevance = relevance

    def get_v(self):
        return self.v

    def get_att(self):
        return self.att

    def get_att_grad(self):
        return self.att_grad

    def get_v_relevance(self):
        return self.v_relevance

    def get_att_relevance(self):
        return self.att_relevance


    def forward(self, x):
        batch, n, embed_size = x.shape
        qkv = self.qkv(x)

        ## ours
        qkv = torch.reshape(qkv, (batch, n, 3, self.n_heads, embed_size//self.n_heads))
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        self.store_v(v)

        # A = Q * K.T
        scaled_products = self.matmul1([q, k]) * self.scale

        att = self.att_softmax(scaled_products)
        att = self.att_dropout(att)

        self.store_att(att)

        att.register_hook(self.store_att_grad)

        # att = A*V
        x = self.matmul2([att, v])
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (batch, n, embed_size))


        x = self.proj(x)
        x = self.out_dropout(x)

        return x

    def relevance_propagation(self, relevance):
        batch, n, embed_size = relevance.shape

        relevance = self.out_dropout.relevance_propagation(relevance)
        relevance = self.proj.relevance_propagation(relevance)

        relevance = torch.reshape(relevance,
                            (batch, n, self.n_heads, embed_size//self.n_heads))
        relevance = relevance.permute(0, 2, 1, 3)


        relevance, relevance_v = self.matmul2.relevance_propagation(relevance)
        ## TODO why? /2
        relevance /=2
        relevance_v /=2

        self.store_v_relevance(relevance_v)
        self.store_att_relevance(relevance)

        relevance = self.att_dropout.relevance_propagation(relevance)
        relevance = self.att_softmax.relevance_propagation(relevance)

        relevance_q, relevance_k = self.matmul1.relevance_propagation(relevance)

        ## TODO why? /2
        relevance_q /=2
        relevance_k /=2

        relevance_qkv = torch.stack([relevance_q,
                                   relevance_k,
                                   relevance_v])

        relevance_qkv = relevance_qkv.permute(1, 3, 0, 2, 4)
        relevance_qkv = torch.reshape(relevance_qkv, (batch, n, 3*embed_size))

        relevance_qkv = self.qkv.relevance_propagation(relevance_qkv)

        return relevance_qkv

class Mlp(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, dropout=0.):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, in_dim)
        self.dropout = Dropout(dropout)
        self.gelu = GELU()

        # TODO rel_prop

    def forward(self, x):
        ## FC1
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        ## FC2
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0.,
                 mlp_hidden_ratio=4):
        super().__init__()

        self.embed_size = embed_size
        self.n_heads = n_heads
        self.QKV_bias = QKV_bias
        self.mlp_hidden_ratio = mlp_hidden_ratio


        ## MLP layer
        self.mlp = Mlp(embed_size, hidden_dim=int(self.mlp_hidden_ratio*self.embed_size), dropout=out_dropout)

        ## Attention layer
        self.attn = Attention_layer(embed_size=self.embed_size, n_heads=self.n_heads, QKV_bias=self.QKV_bias,
                                   att_dropout=att_dropout, out_dropout=out_dropout)

        ## Normalization layers
        self.norm1 = LayerNorm(self.embed_size, eps=1e-6)
        self.norm2 = LayerNorm(self.embed_size, eps=1e-6)

        self.add1 = Add()
        self.add2 = Add()

        self.clone1 = Clone()
        self.clone2 = Clone()



    def forward(self,x):

        x1, x2 = self.clone1(x, 2)
        x2 = self.norm1(x2)
        x2 = self.attn(x2)
        x = self.add1([x1, x2])

        x1, x2 = self.clone2(x, 2)
        x2 = self.norm2(x2)
        x2 = self.mlp(x2)
        x = self.add2([x1, x2])

        return x
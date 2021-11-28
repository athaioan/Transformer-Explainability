import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_divide(a, b) -> object:
    ## stolen from https://github.com/hila-chefer/Transformer-Explainability
    ## Thanks Hila Chefer

    ## avoid dividing with a very low number
    ## retain the sign of each unit

    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())



def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.input = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.input.append(x)
    else:
        self.input = input[0].detach()
        self.input.requires_grad = True

    self.output = output

class RelProp(nn.Module):   # todo --> Stolen from Hila Chefer
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def relevance_propagation(self, R):
        return R

class RelPropSimple(RelProp):    ### todo -> Stolen from Hila Chefer
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class Conv2d(nn.Conv2d, RelProp):

    def test_funct(self):
        print("")

    #TODO REL_PROP

class Add(RelPropSimple): ##  todo --> Change when implementing  Rel_pro

    def forward(self, inputs):
        return torch.add(*inputs)

    ###### NEW ######
    def relevance_propagation(self, relevance):
        sum_relevance = relevance.sum()
        layer_out = self.forward(self.X)
        RL_div = safe_divide(relevance, layer_out)
        gradients = torch.autograd.grad(layer_out, self.X, RL_div, retain_graph=True)
        n_relevances = len(gradients)

        relprop_mul = [self.X[i] * gradients[i] for i in range(n_relevances)]   # out * grad
        relprop_sum = [relprop_mul[i].sum() for i in range(n_relevances)]       # (out * grad).sum
        relprop_abs = [relprop_sum[i].abs() for i in range(n_relevances)]       # |(out * grad).sum|
        relprop_abs_sum = 0
        for i in range(len(relprop_abs)):
            relprop_abs_sum += relprop_abs[i]                                   # SUM[|(out * grad).sum|]

        relprop_out = [safe_divide(relprop_abs[i], relprop_abs_sum) * sum_relevance for i in range(n_relevances)]
        relprop_out = [relprop_mul[i] * safe_divide(relprop_out[i], relprop_sum[i]) for i in range(n_relevances)]

        return relprop_out

class Clone(RelProp): ##  todo --> Change when implementing  Rel_pro

    def forward(self, input, num):
        self.num = num
        clone_list = []
        for _ in range(num):
            clone_list.append(input)

        return clone_list

    ###### NEW ######
    def relevance_propagation(self, relevance):
        layer_out = self.forward(self.X, self.num)
        RL_div = [safe_divide(rel, out) for rel, out in zip(relevance, layer_out)]
        relprop_out = torch.autograd.grad(layer_out, self.X, RL_div, retain_graph=True)
        relprop_out = self.X * relprop_out[0]

        return relprop_out

class Linear(nn.Linear, RelProp):


    def relevance_propagation(self, relevance):

        pos_weights = torch.clamp(self.weight, min=0)
        neg_weights = torch.clamp(self.weight, max=0)

        pos_input = torch.clamp(self.input, min=0)
        neg_input = torch.clamp(self.input, max=0)

        pos_L = F.linear(pos_input, pos_weights)
        neg_L = F.linear(neg_input, neg_weights)

        RL_div_1 = safe_divide(relevance, pos_L + neg_L)
        RL_div_2 = safe_divide(relevance, pos_L + neg_L)

        relevance_1 = pos_input * torch.autograd.grad(pos_L, pos_input, RL_div_1)[0]
        relevance_2 = neg_input * torch.autograd.grad(neg_L, neg_input, RL_div_2)[0]

        relevance = (relevance_1 + relevance_2).unsqueeze(1)

        return relevance

    #TODO REL_PROP

class Matmul(nn.Module):

    def __init__(self, transpose=False):
        super().__init__()
        self.transpose = transpose

    def forward(self, X):
        if self.transpose:
            return torch.matmul(X[0], torch.transpose(X[1], 2, 3))
        else:
            return torch.matmul(X[0], X[1])


    #TODO REL_PROP


class Softmax(nn.Softmax, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class ClsSelect(RelProp):  ## Change when implementing  Rel_pro

    def forward(self, input, dim, index):

        self.dim = dim
        self.index = index

        return torch.index_select(input, dim, index)

    def relevance_propagation(self, relevance):

        L = self.forward(self.input, self.dim, self.index)
        RL_div = safe_divide(relevance, L)
        relevance = torch.autograd.grad(L, self.input, RL_div, retain_graph=True)

        if torch.is_tensor(self.input) == False:
            relevance = []
            relevance.append(self.input[0] * relevance[0])
            relevance.append(self.input[1] * relevance[1])
        else:
            relevance = self.input * (relevance[0])

        return relevance



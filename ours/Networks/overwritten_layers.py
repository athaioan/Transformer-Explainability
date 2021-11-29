import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_divide(a, b) -> object:
    ## stolen from https://github.com/hila-chefer/Transformer-Explainability
    ## Thanks Hila Chefer
    # b[0,0] = -5
    # b[0,1] = 1e-10
    # b[0,2] = -1e-10
    # b[0,3] = 5
    # b[0,4] = 0
    # b = b[0,:5]

    ## avoid dividing with a very low number
    ## retain the sign of each unit

    ## TA:
    # if x >= eps: x
    # if 0<=x<eps: +eps
    # if x<-eps: x
    # if -eps<=x<0: -eps

    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())



def forward_hook(self, input, output):
    ## stolen from https://github.com/hila-chefer/Transformer-Explainability
    ## Thanks Hila Chefer

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


class RelProp(nn.Module):

    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def relevance_propagation(self, relevance):
        return relevance

class RelPropSimple(RelProp):

    def relevance_propagation(self, R):
        L = self.forward(self.input)
        RL_div = safe_divide(R, L)
        relevance = torch.autograd.grad(L,  self.input, RL_div, retain_graph=True)

        if torch.is_tensor(self.input) == False:
            relevance_out = [self.input[index]*relevance[index] for index in range(len(self.input))]
        else:
            relevance_out = self.input * (relevance[0])

        return relevance_out

class Conv2d(nn.Conv2d, RelProp):

    def test_funct(self):
        print("")

    #TODO REL_PROP

class Add(RelPropSimple): ## Change when implementing  Rel_pro

    def forward(self, inputs):
        return torch.add(*inputs)


    ###### GM NEW ###### todo --> remove comment after explaining
    def relevance_propagation(self, relevance):
        relevance_in_sum = relevance.sum()
        layer_out = self.forward(self.input)
        RL_div = safe_divide(relevance, layer_out)
        gradients = torch.autograd.grad(layer_out, self.input, RL_div, retain_graph=True)
        n_relevances = len(gradients)

        relevance_mul = [self.input[i] * gradients[i] for i in range(n_relevances)]   # out * grad
        relevance_sum = [relevance_mul[i].sum() for i in range(n_relevances)]       # (out * grad).sum
        relevance_abs = [relevance_sum[i].abs() for i in range(n_relevances)]       # |(out * grad).sum|
        relevance_abs_sum = 0
        for i in range(len(relevance_abs)):
            relevance_abs_sum += relevance_abs[i]                                   # SUM[|(out * grad).sum|]

        relevance = [safe_divide(relevance_abs[i], relevance_abs_sum) * relevance_in_sum for i in range(n_relevances)]
        relevance = [relevance_mul[i] * safe_divide(relevance[i], relevance_sum[i]) for i in range(n_relevances)]

        return relevance



class Clone(RelProp): ## Change when implementing  Rel_pro

    def forward(self, input, num):

        self.num = num
        clone_list = []
        for _ in range(num):
            clone_list.append(input)

        return clone_list

    def relevance_propagation(self, relevance):
        layer_out = self.forward(self.input, self.num)
        RL_div = [safe_divide(rel, out) for rel, out in zip(relevance, layer_out)]
        relevance = torch.autograd.grad(layer_out, self.input, RL_div, retain_graph=True)
        relevance = self.input * relevance[0]

        return relevance

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

        relevance_1 = pos_input * torch.autograd.grad(pos_L, pos_input, RL_div_1, retain_graph=True)[0]
        relevance_2 = neg_input * torch.autograd.grad(neg_L, neg_input, RL_div_2, retain_graph=True)[0]

        relevance = relevance_1 + relevance_2

        return relevance


class Matmul(RelPropSimple):

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



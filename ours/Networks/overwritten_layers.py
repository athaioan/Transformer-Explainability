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

### Inherit nn.Module wheen possible



class RelProp(nn.Module):
    ## TODO why? /2

    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def relevance_propagation(self, R):
        return R

class RelPropSimple(RelProp):
    ## TODO why? /2

    def relevance_propagation(self, R):
        Z = self.forward(self.input)
        S = safe_divide(R, Z)
        # C = self.gradprop(Z, self.X, S)
        C = torch.autograd.grad(Z,  self.input, S, retain_graph=False)


        if torch.is_tensor(self.input) == False:
            outputs = []
            outputs.append(self.input[0] * C[0])
            outputs.append(self.input[1] * C[1])
        else:
            outputs = self.input * (C[0])
        return outputs

class Conv2d(nn.Conv2d, RelProp):

    def test_funct(self):
        print("")

    #TODO REL_PROP

class Add(nn.Module): ## Change when implementing  Rel_pro

    def forward(self, inputs):
        return torch.add(*inputs)

    def test_funct(self):
        print("")

    #TODO REL_PROP


class Clone(RelProp): ## Change when implementing  Rel_pro

    def forward(self, input, num):

        self.num = num
        clone_list = []
        for _ in range(num):
            clone_list.append(input)

        return clone_list

    #TODO REL_PROP

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



import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):

    def test_funct(self):
        print("")

    #TODO REL_PROP

class Add(nn.Module): ## Change when implementing  Rel_pro

    def forward(self, inputs):
        return torch.add(*inputs)

    def test_funct(self):
        print("")

    #TODO REL_PROP


class Clone(nn.Module): ## Change when implementing  Rel_pro

    def forward(self, input, num):

        self.num = num
        clone_list = []
        for _ in range(num):
            clone_list.append(input)

        return clone_list

    #TODO REL_PROP

class Linear(nn.Linear):


    def test_funct(self):
        print("")

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


class Softmax(nn.Softmax):
    pass

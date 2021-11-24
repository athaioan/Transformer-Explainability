import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):

    def test_funct(self):
        print("")

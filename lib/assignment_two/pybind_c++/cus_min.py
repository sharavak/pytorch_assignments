import sys,torch,os
sys.path.append(os.getcwd()+"/build")
print(os.getcwd())
import custom_min
from torch import nn


class TorchMinimumModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.minimum(a,b)


class TorchDecomposeMinimumModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        obj=custom_min.Cus_Min()
        return obj.cus_min(a,b)

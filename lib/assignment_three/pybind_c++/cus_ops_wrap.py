import torch,sys
from torch import nn
sys.path.append('/home/shar/pytorch_assignments/lib/assignment_three/pybind_c++/build')
import  custom_ops


obj=custom_ops.Custom_Ops_Wrapper()

class CustomOps(nn.Module):

    def __init__(self):
        super().__init__()

    def torch_logaddexp(self,a,b):
        return torch.logaddexp(a,b)

    def torch_addbmm(self,input,a,b):
        return torch.addbmm(input,a,b)

    def forward(self,a,b):
        return obj.cus_logaddexp(a,b)

    def cus_addbmm_wrap(self,input,a,b):
        return obj.cus_addbmm(input,a,b)
    

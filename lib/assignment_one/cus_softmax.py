import torch
from torch import nn


class TorchSoftMaxModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, dim):
        m = nn.Softmax(dim)
        return m(input)


class TorchDecomposeSoftMaxModule(nn.Module):
    def __init__(self):
        super().__init__()

    # 1+x/1!+x^2/2!+x^3/3!+....
    def cus_exponent(self, input):
        fact = 1
        res = torch.ones_like(input)
        x = torch.ones_like(input)
        for i in range(1, 11):
            fact = torch.mul(fact, i)
            x = torch.mul(x, input)
            res = torch.add(res, torch.div(x, fact))
        return res

    def cus_softmax(self, input, dim):
        exp = self.cus_exponent(input)
        exp_sum = torch.sum(exp, dim, keepdim=True)
        return torch.div(exp, exp_sum)

    def forward(self, input, dim):
        return self.cus_softmax(input, dim)

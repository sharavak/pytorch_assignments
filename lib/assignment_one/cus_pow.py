import torch
from torch import nn


class TorchPowerModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, base, exp):
        return torch.pow(base, exp)


class TorchDecomposePowerModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, base, exp):
        return self.torch_cus_pow(base, exp)

    def torch_pow_imp(self, base, exp):
        shape = base.shape
        res = torch.ones(shape, dtype=base.dtype)
        while exp != 0:
            if exp % 2 != 0:
                res = torch.mul(res, base)
            exp = torch.div(exp, 2, rounding_mode="floor")
            base = torch.mul(base, base)
        return res

    def torch_cus_pow(self, base, exp):
        if not torch.is_tensor(exp):
            exp = torch.tensor(exp, dtype=torch.float)
        # For exponent scalar
        if not exp.ndim:
            if exp < 0:
                int_datatypes = set(
                    [
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                        torch.int8,
                        torch.int16,
                        torch.short,
                        torch.long,
                    ]
                )
                if base.dtype == torch.int64 or base.dtype == torch.int32:
                    if exp.dtype in int_datatypes:
                        return torch.div(
                            1, self.torch_pow_imp(base, -exp), rounding_mode="trunc"
                        )
                    return torch.div(1, self.torch_pow_imp(base, -exp))
                return torch.div(1, self.torch_pow_imp(base, -exp))
            return self.torch_pow_imp(base, exp)

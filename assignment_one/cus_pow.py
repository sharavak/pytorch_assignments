import torch


def torch_pow(base, exp):
    shape = base.shape
    res = torch.ones(shape, dtype=base.dtype)
    # For exponent scalar
    if not exp.ndim:
      while exp != 0:
          if exp % 2 != 0:
              res = torch.mul(res, base)
          exp = torch.div(exp, 2, rounding_mode="floor")
          base = torch.mul(base, base)
    return res


def torch_cus_pow(base, exp):
    if exp < 0:
        int_datatypes=set(torch.int32,torch.int64,torch.uint8,torch.int8,torch.int16,torch.short,torch.long)
        exp=torch.tensor(exp,dtype=torch.float)
        if  base.dtype==torch.int64 or base.dtype==torch.int32:
            return torch.div(1, torch_pow(base, -exp), rounding_mode="trunc")
        return torch.div(1, torch_pow(base, -exp))
    return torch_pow(base, exp)

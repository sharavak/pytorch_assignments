import torch


def torch_pow(base, exp):
    shape = base.shape
    res = torch.ones(shape, dtype=base.dtype)
    while exp != 0:
        if exp % 2 != 0:
            res = torch.mul(res, base)
        exp = torch.div(exp, 2, rounding_mode="floor")
        base = torch.mul(base, base)
    return res


def torch_cus_pow(base, exp):
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
                    return torch.div(1, torch_pow(base, -exp), rounding_mode="trunc")
                return torch.div(1, torch_pow(base, -exp))
            return torch.div(1, torch_pow(base, -exp))
        return torch_pow(base, exp)
    else:
        return expo_tensor(base,exp)

def expo_tensor(base,exp):
    b1 = base.reshape(-1)
    d1 = base.shape
    d2 = exp.shape
    m, n = len(d1), len(d2)
    dims = [1] * max(m, n)
    d1 = list(d1)
    d2 = list(d2)
    if len(d1) < len(d2):
        d1 = [1] * (n - m) + d1
        for i in range(max(m, n)):
            dims[i] = max(d1[i], d2[i])
        exp_tp = base.reshape(dims)
        b1 = base.reshape(-1)
        e1 = exp_tp.reshape(-1)
    else:
        d2 = [1] * (m - n) + d2
        for i in range(max(m, n)):
            dims[i] = max(d1[i], d2[i])
        exp_tp = exp.reshape(dims)
        b1 = base.reshape(-1)
        e1 = exp_tp.reshape(-1)

    res = torch.ones(b1.shape, dtype=base.dtype)

    for i in range(e1.shape[0]):
        res[i] = torch_cus_pow(b1[i], e1[i])
    return res

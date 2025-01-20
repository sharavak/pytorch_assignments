import pytest, torch
import sys

sys.path.append("/home/shar/pytorch_assignments")
from assignment_one.cus_pow import torch_cus_pow

test_suite = []
for i in range(5):
    base = torch.randn(1, 4)
    if i % 2 == 0:
        base, exp = torch.tensor([8, 9, 8]), torch.tensor((i + 1))
    test_suite.append([base, exp, torch.pow(base, exp)])


@pytest.mark.parametrize("base,exp,expected", test_suite)
def test_Pow(base, exp, expected):
    assert torch.equal(torch_cus_pow(base,exp),expected)


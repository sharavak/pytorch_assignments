import pytest, torch, os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assignment_one.cus_pow import torch_cus_pow

test_suite = []
for i in range(15):
    base = torch.randn(1, 5)
    exp = 0
    if i % 2 == 0:
        base, exp = torch.tensor([8, 9, 8]), torch.tensor(-(i + 1))
    test_suite.append([base, exp, torch.pow(base, exp)])
test_suite.append(
    [torch.tensor([0, 0, 0, 0]), 2, torch.pow(torch.tensor([0, 0, 0, 0]), 2)]
)

# For exponent tensor when they have the dimension is 1

base = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
exp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

test_suite.append([base, exp, torch.pow(base, exp)])


@pytest.mark.parametrize("base,exp,expected", test_suite)
def test_pow(base, exp, expected):
    assert torch.equal(torch_cus_pow(base, exp), expected)

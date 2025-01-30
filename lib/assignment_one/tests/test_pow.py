import pytest, torch, os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.assignment_one.cus_pow import TorchPowerModule, TorchDecomposePowerModule

t_pow = TorchPowerModule()
d_pow = TorchDecomposePowerModule()

test_suite = []
shapes = [[1, 2], [1, 2, 3], [5, 5], [1, 1, 2, 2], [1, 1]]
for i in range(5):
    base = torch.randn(shapes[i])
    exp = i
    test_suite.append([base, exp, t_pow.forward(base, exp)])

exp_tens=torch.tensor([1,2,3])

test_suite.append([torch.tensor([8,6,7]),exp_tens,t_pow.forward(torch.tensor([8,6,7]),exp_tens)])

test_suite.append(
    [torch.tensor([0, 0, 0, 0]), 2, t_pow.forward(torch.tensor([0, 0, 0, 0]), 2)]
)


@pytest.mark.parametrize("base,exp,expected", test_suite)
def test_pow(base, exp, expected):
    assert torch.allclose(d_pow.forward(base, exp), expected)

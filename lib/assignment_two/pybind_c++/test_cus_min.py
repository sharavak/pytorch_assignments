import pytest, torch, os, random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cus_min import TorchMinimumModule,TorchDecomposeMinimumModule

t_min = TorchMinimumModule()
d_min = TorchDecomposeMinimumModule()

test_suite = []
shapes = [[1, 2], [1, 2, 3], [5, 5], [2, 1, 2, 2], [1, 1]]

for i in range(5):
    a = torch.randint(1,1000,shapes[i],dtype=torch.float32)
    b = torch.randint(1,1000,shapes[i],dtype=torch.float32)

    test_suite.append([a, b, t_min.forward(a, b)])






@pytest.mark.parametrize("a,b,expected", test_suite)
def test_softmax(a, b, expected):
    assert torch.close(d_min.forward(a, b), expected)
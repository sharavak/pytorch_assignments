import pytest, torch, os, random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.assignment_one.cus_softmax import (
    TorchSoftMaxModule,
    TorchDecomposeSoftMaxModule,
)

t_softmax = TorchSoftMaxModule()
d_softmax = TorchDecomposeSoftMaxModule()

test_suite = []
shapes = [[1, 2], [1, 2, 3], [5, 5], [2, 1, 2, 2], [1, 1]]

for i in range(5):
    input = torch.randint(1,1000,shapes[i],dtype=torch.float32)
    dim = random.choice(list(range(0, input.ndim - 1)))
    test_suite.append([input, dim, t_softmax.forward(input, dim)])




inp = torch.tensor([[1, 2, 3], [1, 2, 3]])
test_suite.append([inp, 0, t_softmax.forward(inp, 0)])


@pytest.mark.parametrize("input,dim,expected", test_suite)
def test_softmax(input, dim, expected):
    assert torch.allclose(d_softmax.forward(input, dim), expected)

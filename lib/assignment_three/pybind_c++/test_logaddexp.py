import pytest, torch
from cus_ops_wrap import CustomOps


obj=CustomOps()
test_suite1 = []
shapes = [[1, 2], [2,2,5], [16,16], [5,8], [8,9]]


for i in range(5):
    a = torch.randn(shapes[i])
    b = torch.randn(shapes[i])

    test_suite1.append([a, b, obj.torch_logaddexp(a, b)])

@pytest.mark.parametrize("a,b,expected", test_suite1)
def test_logaddexp(a, b, expected):
    assert torch.allclose(obj.forward(a, b), expected)


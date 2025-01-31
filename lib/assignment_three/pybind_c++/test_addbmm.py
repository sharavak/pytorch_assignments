import pytest, torch
from cus_ops_wrap import CustomOps

obj=CustomOps()

test_suite2 = []
shapes1=[[1,2,3],[5,8,3],[2,4,9]]
shapes2=[[1,3,5],[5,3,7],[2,9,10]]
for i in range(len(shapes1)):
    a = torch.randn(shapes1[i]) # b1 x m x n
    b = torch.randn(shapes2[i]) # b1 x n x p
    r1=a.shape[1]
    c2=b.shape[2]
    input = torch.randn(r1,c2) # m x p
    test_suite2.append([input,a,b,obj.torch_addbmm(input,a,b)])


@pytest.mark.parametrize("input,a,b,expected", test_suite2)
def test_logaddexp(input ,a, b, expected):
    assert torch.allclose(obj.cus_addbmm_wrap(input,a, b), expected)
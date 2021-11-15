import torch
import numpy

input = torch.randn(10, 3, 4)
mat2 = torch.randn(10, 4, 5)
res = torch.bmm(input, mat2)
print(res.size())

in0 = input[0]
mat0 = mat2[0]
res0 = torch.matmul(in0, mat0)
print(res0.size())

diff = torch.subtract(res[0], res0)
equal = torch.equal(res[0], res0)
print(numpy.max(diff.cpu().numpy()), equal)
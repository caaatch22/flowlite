import sys
sys.path.append('..')
import pytensor as pt
import pytensor.nn as nn
# import needle as ndl

x = pt.Tensor([1, 3, 2], dtype='float32', requires_grad=True)
y = 2 * x
y.backward()
print(x.grad)

a = pt.ones(3, 4, dtype='int32')
print(f'a : {a}')

b = a.sum(1, keepdim=True)
print(f'b : {b}')

z = pt.randn(2, 4, dtype='float32')
# zz = z.sum()
# print(f'zz : {zz.__repr__}')

aa = pt.randn(3, 4, requires_grad=True)
print(f'aa : {aa.__dict__}')
bb = nn.Parameter(aa)
print(f'bb : {bb.__dict__}')

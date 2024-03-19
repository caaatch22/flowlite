import sys
sys.path.append('..')
import pytensor as pt
# import needle as ndl

x = pt.Tensor([1, 3, 2], dtype='float32', requires_grad=True)
y = 2 * x
y.backward()
print(x.grad)



z = pt.randn(2, 4, dtype='float32')
zz = z.sum()
print(f'zz : {zz.__repr__}')
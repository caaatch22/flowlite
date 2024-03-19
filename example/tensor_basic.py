import sys
sys.path.append('..')
import pytensor as pt
# import needle as ndl

x = pt.Tensor([1, 3, 2], dtype='float32')
y = x + 1

y = y + 1
print(y.grad)
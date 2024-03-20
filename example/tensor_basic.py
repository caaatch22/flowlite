import flowlite as fl
import flowlite.nn as nn

x = fl.randn(2, 4, 5)
y = fl.randn(2, 4)
x = fl.ones(2, 3)


print(f'{x.dtype}')

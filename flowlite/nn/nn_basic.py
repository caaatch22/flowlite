from typing import List
from flowlite.autograd import Tensor
from flowlite import ops
import flowlite.init as init


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""
    def __init__(self, x: Tensor):
        super().__init__(x, requires_grad=True)


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

def _child_modules(value: object) -> List['Module']:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []

class Module:
    def __init__(self):
        self.training = True
    
    def parameters(self) -> List[Tensor]:
        return _unpack_params(self.__dict__)

    def _children(self) -> List['Module']:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False
    
    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose(0, 1))
        else:
            bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out

class ReLU(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x, inplace=self.inplace)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        from functools import reduce
        return reduce(lambda out, module: module(out), self.modules, x)

class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        from functools import reduce
        size = reduce(lambda a, b: a * b, x.shape)
        # we assume the first dim is batch_size
        return x.reshape((x.shape[0], size // x.shape[0]))


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        '''
        logits: (bs, num_classes), 
        '''
        one_hot_y = init.one_hot(logits.shape[1], y, device=logits.device)
        return ops.sum(ops.logsumexp(logits, axes=1) - (logits * one_hot_y).sum(dim=1)) / logits.shape[0]


# TODO: inplace
class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p)
            return x * mask / (1 - self.p)
        else:
            return x

#TODO: all the device and dtype parameters in bn and ln
class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(num_features))
        self.bias = Parameter(init.zeros(num_features))
        self.running_mean = init.zeros(num_features)
        self.running_var = init.ones(num_features)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, C) C is #feature or #channels
        if self.training:
            mean = x.sum(dim=0) / x.shape[0] # (num_features, )
            var = ((x - mean.broadcast_to(x.shape))**2).sum(dim=0) / x.shape[0] #(num_features,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            norm = (x - mean.broadcast_to(x.shape)) / ((var.broadcast_to(x.shape) + self.eps) ** 0.5)
            #TODO: auto broadcast_to support in numpy but not in our implemented NDArray
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)

class LayerNorm1d(Module):
    def __init__(self, num_features: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        
        self.weight = Parameter(init.ones(num_features))
        self.bias = Parameter(init.zeros(num_features))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, C)
        mean = x.sum(dim=1, keepdim=True).broadcast_to(x.shape) / x.shape[1]
        var = ((x - mean)**2).sum(dim=1, keepdim=True).broadcast_to(x.shape) / x.shape[1]
        norm = (x - mean) / ((var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
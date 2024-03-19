from typing import List, Callable, Any
from pytensor.autograd import Tensor
from pytensor import ops
import pytensor.init as init


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


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
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).transpose(0, 1))
        else:
            bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out

class ReLU(Module):
    # TODO: about inplace=True
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.module = modules

    def forward(self, x: Tensor) -> Tensor:
        from functools import reduce
        return reduce(lambda out, module: module(out), self.module, x)

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
        one_hot_y = init.one_hot(logits.shape[1], y)
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

class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(num_features, requires_grad=True))
        self.bias = Parameter(init.zeros(num_features, requires_grad=True))
        self.running_mean = init.zeros(num_features)
        self.running_var = init.ones(num_features)
        
    def forward(x: Tensor) -> Tensor:
        # x: (batch_size, C) C is #feature or #channels
        pass

class LayerNorm1d(Module):
    pass

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
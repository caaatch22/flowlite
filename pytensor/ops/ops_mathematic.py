
from typing import Optional, Union, Tuple

from ..autograd import NDArray
from ..autograd import Op, Tensor

import numpy as array_api

class EWiseAdd(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad, out_grad
    
def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(Op):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)         


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(Op):
    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = out_grad * (-a / (b**2))
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(Op):
    def __init__(self, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1

    def compute(self, x: NDArray) -> NDArray:
        return array_api.swapaxes(x, self.dim0, self.dim1)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.dim0, self.dim1)


def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    return Transpose(dim0=dim0, dim1=dim1)(x)


# TODO: to be consistent with pytorch
class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        return x.reshape(self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Sum(Op):
    def __init__(self, dim: Union[int, Tuple[int, ...]] = None, keepdim: bool = False):
        self.dim = dim
        self.keepdim = keepdim

    def compute(self, a):
        return array_api.sum(a, self.dim)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        #TODO: not sure for keepdim == True
        if self.keepdim == True:
            return out_grad.broadcast_to(node.inputs[0].shape)

        new_shape = list(node.inputs[0].shape)
        dims = range(len(new_shape)) if self.dim is None else self.dim
        if isinstance(self.dim, int):
            dims = [self.dim]  # Convert single integer to a list
        for dim in dims:
            new_shape[dim] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)


def sum(x: Tensor, dim = None, keepdim: bool = False):
    return Sum(dim, keepdim)(x)


class MatMul(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_ndim = len(lhs.shape)
        rhs_ndim = len(rhs.shape)
        #TODO: maybe let transpose accept None
        lgrad, rgrad = matmul(out_grad, rhs.transpose(rhs_ndim - 1, rhs_ndim - 2)), matmul(lhs.transpose(lhs_ndim - 1, lhs_ndim - 2), out_grad)
        
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(Op):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(Op):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(Op):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


#TODO: add inplace=True
class ReLU(Op):
    # def __init__(inplace: bool = False):

    def compute(self, a):
        out = a.copy()
        out[out < 0] = 0
        return out

    def gradient(self, out_grad, node):
        out = node.underly().copy()
        out[out > 0] = 1
        out[out <= 0] = 0
        return out_grad * Tensor(out)


def relu(a):
    return ReLU()(a)

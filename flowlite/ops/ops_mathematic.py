
from typing import Optional, Union, Tuple

from ..autograd import Op, Tensor

from ..backend_selection import array_api, NDArray

#TODO: for all ops, maybe super().__init()__ in __init__
class EWiseAdd(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad, out_grad
    
def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class AddScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


def add_scalar(a: Tensor, scalar) -> Tensor:
    return AddScalar(scalar)(a)


class EWiseMul(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class MulScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return (out_grad * self.scalar,)


def mul_scalar(a: Tensor, scalar) -> Tensor:
    return MulScalar(scalar)(a)


class PowerScalar(Op):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)         


def power_scalar(a: Tensor, scalar) -> Tensor:
    return PowerScalar(scalar)(a)


class EWisePow(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class EWiseDiv(Op):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = out_grad * (-a / (b**2))
        return grad_a, grad_b


def divide(a: Tensor, b: Tensor):
    return EWiseDiv()(a, b)


class DivScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar


def divide_scalar(a: Tensor, scalar) -> Tensor:
    return DivScalar(scalar)(a)


class Transpose(Op):
    def __init__(self, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1

    def compute(self, x: NDArray) -> NDArray:
        return array_api.swapaxes(x, self.dim0, self.dim1)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad.transpose(self.dim0, self.dim1)


def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    return Transpose(dim0=dim0, dim1=dim1)(x)


# TODO: to be consistent with pytorch
class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        return array_api.reshape(x, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad.reshape(node.inputs[0].shape)


def reshape(x: Tensor, shape):
    return Reshape(shape)(x)


class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        return array_api.broadcast_to(x, self.shape).compact()
    
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        #TODO: not sure, but should be this
        # when shrink_dim == (), means that in forward path, we broadcasted to the same shape
        # for example a.broadcast_to(x.shape) when x.shape == (3, 1) and a.shape == (3, 1)
        # test_nn_layernorm_backward_4
        if shrink_dims == ():
            return out_grad
        return out_grad.sum(shrink_dims).reshape(ori_shape)


def broadcast_to(x: Tensor, shape):
    return BroadcastTo(shape)(x)


class Sum(Op):
    def __init__(self, dim: Union[int, Tuple[int, ...]] = None, keepdim: bool = False):
        self.dim = dim
        self.keepdim = keepdim

    def compute(self, x) -> NDArray:
        return array_api.sum(x, dim=self.dim, keepdim=self.keepdim)

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

    def gradient(self, out_grad: Tensor, node: Tensor):
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


def matmul(a: Tensor, b: Tensor) -> Tensor:
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


#TODO: use inplace
class ReLU(Op):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def compute(self, x: NDArray) -> NDArray:
        return array_api.maximum(x, 0)

    #TODO: [maybe BUG]: maybe deep copy of node.unerly() ?
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        out = node.underly()
        return out_grad * Tensor(out > 0, device=out_grad.device)


def relu(x, inplace: bool = False):
    return ReLU(inplace)(x)

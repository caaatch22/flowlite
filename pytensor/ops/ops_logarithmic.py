from typing import Optional
from ..autograd import Op, Tensor
from .ops_mathematic import *
import numpy as array_api


class LogSumExp(Op):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: Tensor):
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduce = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z_reduce

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        max_z = z.underly().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = sum(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        if isinstance(axes, int):
            axes = [axes]
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z


def logsumexp(input, axes=None):
    return LogSumExp(axes=axes)(input)

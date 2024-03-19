import pytensor
from .backend_numpy import Device, cpu, all_devices
from typing import List, Optional, Dict, Tuple, Union
import numpy

from pytensor import init

LAZY_MODE = False
TENSOR_COUNTER = 0

import numpy as array_api

NDArray = numpy.ndarray


class Op:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

    def compute(self, *args: Tuple[NDArray]) -> NDArray:
        raise NotImplementedError()
    
    def gradient(self, out_grad: 'Tensor', node: 'Tensor') -> Union['Tensor', Tuple['Tensor']]:
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: 'Tensor', node: 'Tensor') -> Tuple['Tensor']:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class Tensor:
    op: Optional[Op]
    inputs: List['Tensor']
    shared_data: NDArray
    requires_grad: bool
    grad: 'Tensor'
    
    def __init__(
        self,
        array: Union['Tensor', NDArray],
        *,
        device: Optional[Device] = None,
        dtype = None,
        requires_grad: bool = False,
        **kwargs
    ) -> None:
        if isinstance(array, Tensor):
            device = array.device if device is None else device
            dtype = array.dtype if dtype is None else dtype
            if device == array.device and dtype == array.detype:
                shared_data = array.realize_data()
            # TODO: _array_from_numpy should be in NDArray
            else:
                shared_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        
        else:
            device = device if device else cpu()
            shared_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        
        self._init(None, [], shared_data=shared_data, requires_grad=requires_grad)
    
    def _init(self, op: Optional[Op], inputs: List['Tensor'], *, 
              num_outputs: int = 1, shared_data = None, requires_grad: Optional[bool] = None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.shared_data = shared_data
        self.requires_grad = requires_grad
        
    
    @staticmethod
    def _array_from_numpy(numpy_array: numpy.ndarray, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Op, inputs: List['Tensor']) -> 'Tensor':
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_data()
        return tensor
    
    @staticmethod
    def make_const(data: NDArray, requires_grad: bool = False) -> 'Tensor':
        """Create a new tensor that shares the data but detaches from the graph."""
        '''Used in detach()'''
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            shared_data=data if not isinstance(data, Tensor) else data.realize_data(),
            requires_grad=requires_grad,
        )
        return tensor    

    
    def realize_data(self) -> NDArray:
        if self.shared_data is not None:
            return self.shared_data
        # if self.data is None, self shouldn't be leaf node which means its op shouldn't be None
        self.shared_data = self.op.compute(
            *[x.realize_data() for x in self.inputs]
        )
        return self.shared_data

    @property
    def data(self):
        return self.detach()
    
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f'{value.dtype} {self.dtype}'        
        self.data = value.realize_data() 

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_data())

    @property
    def shape(self):
        return self.realize_data().shape
    
    @property
    def dtype(self):
        return self.realize_data().dtype
    
    @property
    def device(self):
        data = self.realize_data()
        if array_api is numpy:
            return cpu()
        return data.device


    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def is_leaf(self) -> bool:
        return self.op is None
    
    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1
    
    def __repr__(self) -> str:
        return f'pytensor.Tensor({str(self.realize_data())})'
    
    def __str__(self) -> str:
        return self.realize_data().__str__()
    
    def numpy(self):
        data = self.realize_data()
        if array_api is numpy:
            return data
        return data.numpy()
    

    def __add__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWiseAdd()(self, other)
        else:
            return pytensor.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWiseMul()(self, other)
        else:
            return pytensor.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWisePow()(self, other)
        else:
            return pytensor.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWiseAdd()(self, pytensor.ops.Negate()(other))
        else:
            return pytensor.ops.AddScalar(-other)(self)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWiseAdd()(pytensor.ops.Negate()(self), other)
        else:
            return pytensor.ops.AddScalar(other)(-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return pytensor.ops.EWiseDiv()(self, other)
        else:
            return pytensor.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return pytensor.ops.MatMul()(self, other)

    def matmul(self, other):
        return pytensor.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return pytensor.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return pytensor.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return pytensor.ops.Reshape(shape)(self)

    def __neg__(self):
        return pytensor.ops.Negate()(self)

    def transpose(self, axes=None):
        return pytensor.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
        
        
def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        # sum up partial ajoints
        ajoint = sum_node_list(node_to_output_grads_list[node])
        node.grad = ajoint
        if node.op is None:
            # Leaf node
            continue
        # compute partial ajoints for input node
        partial_ajoints = node.op.gradient_as_tuple(ajoint, node)
        for in_node, partial_ajoint in zip(node.inputs, partial_ajoints):
            if in_node not in node_to_output_grads_list:
                node_to_output_grads_list[in_node] = []
            node_to_output_grads_list[in_node].append(partial_ajoint)



def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    for node in node_list:
        if node not in visited:
            topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    for input_node in node.inputs:
        if input_node not in visited:
            topo_sort_dfs(input_node, visited, topo_order)
    visited.add(node)
    topo_order.append(node)



def sum_node_list(node_list: List[Tensor]):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
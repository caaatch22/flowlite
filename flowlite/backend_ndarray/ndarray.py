import numpy as np
import math
from typing import Tuple, Optional, Union

from . import ndarray_backend_numpy
from . import ndarray_backend_cpu

class BackendDevice:
    """A backend device, wraps the implementation module."""
    def __init__(self, name: str, mod):
        self.name = name
        self.mod = mod
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'({self.name})'
    
    def __getattr__(self, name):
        return getattr(self.mod, name)
    
    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr

def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda        
        return BackendDevice('cuda', ndarray_backend_cuda)
    except ImportError:
        return BackendDevice('cuda', None)

def cpu_numpy():
    """Return numpy device"""
    return BackendDevice('cpu_numpy', ndarray_backend_numpy)

def cpu():
    """Return cpu device"""
    return BackendDevice('cpu', ndarray_backend_cpu)
 
def default_device():
    return cpu_numpy()

def all_devices():
    return [cpu(), cpu_numpy(), cuda()]



class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """
    
    def __init__(self, other, device=None):
        if isinstance(other, NDArray):
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            # for NDArray([1, 2, 3], device=cpu())
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: 'NDArray'):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle
    
    @staticmethod
    def compact_strides(shape: Tuple[int]) -> Tuple[int]:
        '''Utility function to compute compact strides from shape.'''
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])
        
    
    @staticmethod
    def make(shape: Tuple[int], 
             strides: Optional[Tuple[int]] = None, 
             device: Optional[BackendDevice] = None, 
             handle = None, # device.Array 
             offset: int = 0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(math.prod(shape))
        else:
            array._handle = handle
        return array
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def strides(self):
        return self._strides 
    
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):
        return math.prod(self._shape)
    
    def __repr__(self):
        return f'NDArray({self.numpy().__str__()}, device={self.device}'
    
    def __str__(self):
        return self.numpy().__str__()


    def fill(self, value):
        self._device.fill(self._handle, value)
        
    
    def to(self, device: BackendDevice):
        """ Convert between devices, using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)
    
    def numpy(self):
        return self._device.to_numpy(
            self._handle, self._shape, self._strides, self._offset
        )
        
    def is_compact(self):
        return (
            self._strides == self.compact_strides(self._shape)
            and math.prod(self.shape) == self._handle.size
        )
    
    def compact(self):
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self._shape, self._strides, self._offset
            )
            return out
    
    def as_strided(self, shape: Tuple[int], strides: Tuple[int]):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )
    
    @property
    def flat(self):
        return self.reshape((self.size, ))
    
    def reshape(self, dim: Tuple[int]):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """
        
        # handle the case where we have one -1 in the dim:
        assert all(d > 0 or d == -1 for d in dim), "Invalid dimension in reshape"
        assert dim.count(-1) <= 1, "Can only specify one unknown dimension"
        pos = [i for i, d in enumerate(dim) if d > 0]
        total_size = math.prod(self._shape)
        total_pos_size = math.prod([dim[i] for i in pos])
        if total_size % total_pos_size != 0:
            raise ValueError(f"Cannot reshape array of size {self.size} to {dim}, as size doesn't match")
        
        if len(pos) == len(dim) - 1:
            dim = tuple([total_size // total_pos_size if d == -1 else d for d in dim])
        
        
        if math.prod(self._shape) != math.prod(dim):
            raise ValueError(f"Cannot reshape array of size {self.size} to {dim}")
        if not self.is_compact():
            raise ValueError("Cannot reshape non-compact array")
        return self.as_strided(dim, self.compact_strides(dim))
        
    def permute(self, dim: Tuple[int]):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        if len(dim) != self.ndim:
            raise ValueError(f"Cannot permute array with {self.ndim} dimensions to {dim}")
        if not self.is_compact():
            raise ValueError("Cannot permute non-compact array")
        
        new_shape = tuple(self._shape[i] for i in dim)
        new_strides = tuple(self._strides[i] for i in dim)
        return self.as_strided(new_shape, new_strides)
    
    def broadcast_to(self, new_shape: Tuple[int]):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        assert len(self._shape) <= len(new_shape)
        a = self
        if len(self._shape) < len(new_shape):
            assert self._shape == new_shape[len(self._shape):], 'Cannot broadcast to shape with different tailing dimensions'
            a = self.compact().reshape((1, ) * (len(new_shape) - len(self._shape)) + self._shape)
        
        for x, y in zip(a.shape, new_shape):
            assert x == y or x == 1
        
        new_strides = list(a.strides)
        for idx, (x, y) in enumerate(zip(a.shape, new_shape)):
            if x != y:
                new_strides[idx] = 0
        
        return a.as_strided(new_shape, tuple(new_strides))

    ### Get and set elements

    def process_slice(self, sl, dim):
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)
    
    def __getitem__(self, idxs: Union[int, slice, Tuple[int, ...]]):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"
        
        offset = 0
        new_strides = []
        new_shape = []
        for i in range(len(idxs)):
            new_strides.append(self.strides[i] * idxs[i].step)
            offset += self.strides[i] * idxs[i].start
            new_shape.append((idxs[i].stop - idxs[i].start + idxs[i].step - 1 ) // idxs[i].step)

        new_shape = tuple(new_shape)
        new_strides = tuple(new_strides)

        return self.make(shape=new_shape, strides=new_strides, device=self.device, handle=self._handle, offset=offset)
    
    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert math.prod(view.shape) == math.prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                math.prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
    
    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )
    
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)
    
    def __pow__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_power, self.device.scalar_power
        )

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )
    
    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)
    
    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out
    
    def __matmul__(self, other: 'NDArray'):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]
        
        m, n, p = self.shape[0], self.shape[1], other.shape[1]
        
        if hasattr(self.device, 'matmul_tiled') and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):
            def tile(a: 'NDArray', tile_size: int):
                return a.as_strided(
                    (a.shape[0] // tile_size, a.shape[1] // tile_size, tile_size, tile_size),
                    (a.shape[1] * tile_size, tile_size, a.shape[1], 1)
                )
            
            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)
        
            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )
        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out
    
    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, dim: Union[Optional[int], Tuple[int]] = None, keepdim: bool = False):
        """ Return a view to the array set up for reduction functions and output array. """
        #TODO [maybe BUG]: the behavior when dim is empty tuple
        if (isinstance(dim, tuple) and not dim) or dim is None:
            view = self.compact().reshape((1, ) * (self.ndim - 1) + (math.prod(self.shape), ))
            out = NDArray.make((1, ) * (self.ndim if keepdim else 1), device=self.device)
            
        else:
            #TODO: support multiple dim reduction
            if isinstance(dim, tuple):
                assert len(dim) == 1, 'Only support single axis reduction'
                dim = dim[0]
            
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != dim]) + (dim, )
            )
            out = NDArray.make(
                tuple([1 if i == dim else s for i, s in enumerate(self.shape)])
                if keepdim else
                tuple([s for i, s in enumerate(self.shape) if i != dim]),
                device=self.device,
            )
        return view, out
    
    def sum(self, dim: Union[Optional[int], Tuple[int]] = None, keepdim=False):
        view, out = self.reduce_view_out(dim, keepdim=keepdim)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, dim: Union[Optional[int], Tuple[int]] = None, keepdim=False):
        view, out = self.reduce_view_out(dim, keepdim=keepdim)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        assert len(axes) <= len(self.shape)
        new_strides = list(self.strides)
        for axis in axes:
            new_strides[axis] = - new_strides[axis]
        new_strides = tuple(new_strides)
        new_offset = sum([(self.shape[axis] - 1) * self.strides[axis] for axis in axes])
        return NDArray.make(self.shape, new_strides, self._device, self._handle, new_offset).compact()


    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        assert len(axes) == len(self.shape)
        new_shape = tuple([l + r + n for (l, r), n in zip(axes, self.shape)])
        arr = self.device.full(new_shape, 0)
        access = tuple([slice(l, l + n) for (l, _), n in zip(axes, self.shape)])
        arr[access] = self
        return arr



def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def max(a, dim, keepdim=False):
    return a.max(dim, keepdim=keepdim)

def maximum(a, b):
    return a.maximum(b)

def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()

def flip(a, axes):
    return a.flip(axes)


def sum(a, dim=None, keepdim=False):
    return a.sum(dim=dim, keepdim=keepdim)

def swapaxes(x: NDArray, axis1, axis2):
    permute_axes = list(range(x.ndim))
    permute_axes[axis1], permute_axes[axis2] = axis2, axis1
    return x.permute(permute_axes)

def matmul(a: NDArray, b: NDArray):
    return a @ b
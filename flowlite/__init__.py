from . import ops
from .ops import *
from .autograd import Tensor, cpu, all_devices
from .backend_selection import *

from . import init
from .init import ones, zeros, zeros_like, ones_like, randn

from . import nn
from . import optim
from . import data
import math
from .init_basic import *

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std=std, **kwargs)   

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity='relu', **kwargs):
    assert nonlinearity == 'relu', 'Only relu supported currently'
    bound = math.sqrt(6 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity='relu', **kwargs):
    assert nonlinearity == 'relu', 'Only relu supported currently'
    bound = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, std=bound, **kwargs)
import flowlite as fl
import flowlite.nn as nn


class Optimizer:
    def __init__(self, params: nn.Parameter):
        self.params = params
    
    def step(self):
        raise NotImplementedError()
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params: nn.Parameter, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        '''
        u_{t+1} = \beta u_t + (1 - \beta) \nabla_{\theta}f(\theta_t)
        \theta_{t+1} = \theta_t - \alpha u_{t+1}
        
        \alpha: lr
        \beta: momentum
        '''
        # https://discuss.pytorch.org/t/how-does-sgd-weight-decay-work/33105/2
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            grad = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * grad
            #TODO: figure out if it should be ndl.Tensor(grad, dtype=param.dtype, requires_grad=False)
            self.u[param] = fl.Tensor(grad, dtype=param.dtype)
            param.data -= self.lr * self.u[param]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # self.t += 1
        # for w in self.params:
        #     grad = w.grad.data + self.weight_decay * w.data
        #     self.m[w] = self.beta1 * self.m.get(w, 0) + (1 - self.beta1) * grad
        #     self.v[w] = self.beta2 * self.v.get(w, 0) + (1 - self.beta2) * (grad ** 2)
        #     unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
        #     unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
        #     w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)
        self.t += 1
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad ** 2)
            m_hat = (self.m[param] / (1 - self.beta1 ** self.t)).detach()
            v_hat = (self.v[param] / (1 - self.beta2 ** self.t)).detach()
            update = fl.Tensor(self.lr * m_hat / (v_hat ** 0.5 + self.eps), dtype=param.dtype).detach()
            param.data -= update.detach()

#TODO: more optim, like AdaGrad, RMSprop
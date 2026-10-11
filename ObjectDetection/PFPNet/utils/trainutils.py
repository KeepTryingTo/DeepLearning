import time
import torch
import torch.optim as optim


class Timer(object):
    def __init__(self, indic='s'):
        if indic == 'm':
          self.div = 60
        elif indic == 'h':
            self.div = 60 * 60
        else:
            self.div = 1
        self.time = -1

    def tic(self):
        self.time = time.time()

    def toc(self):
        num = (time.time() - self.time) / self.div
        self.time = time.time()
        return num


class LossCal(object):
    def __init__(self):
        self.loc = 0
        self.conf = 0
        self.cnt = 0

    def pop(self):
        try:
            return self.loc / self.cnt, self.conf/ self.cnt
        except ZeroDivisionError:
            print('ERROR: Division by zero')

    def stack(self, loc, conf):
        self.loc += loc
        self.conf += conf
        self.cnt += 1

    def reset(self):
        self.__init__()


class CaffeSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(CaffeSGD, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.sub_(d_p)

        return loss


class CaffeScheduler(object):
    def __init__(self, optimizer, milestones, base_lr, gamma):
        self.optimizer = optimizer
        self.milestones = milestones
        self.base_lr = base_lr
        self.gamma = gamma
        self.iters = 0

    def step(self):
        self.iters += 1
        if self.iters in self.milestones:
            self.base_lr *= self.gamma

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lr * param_group['lr_mult']

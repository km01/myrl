import torch
import numpy as np
import math
import torch.nn.functional as F


def log_normal(x, loc, scale):
    var = scale.pow(2.)
    return -0.5 * (torch.log(2. * math.pi * var) + (x - loc).pow(2.) / var)


class Policy(object):
    def __init__(self, param):
        self.param = param

    def log_prob(self, x, param_grad=True):
        raise NotImplementedError

    def sample(self, det=False):
        raise NotImplementedError

    def as_env(self, act):
        raise NotImplementedError


class Gaussian(Policy):
    def __init__(self, param):
        super().__init__(param)
        self.loc, self.scale = self.param.chunk(2, dim=-1)
        self.scale = F.softplus(self.scale)

    def sample(self, det=False):

        if det:
            x = self.loc

        else:
            noise = self.scale * torch.randn_like(self.scale)
            x = self.loc + noise

        return x

    def log_prob(self, x, param_grad=True):
        if param_grad:
            log_px = log_normal(x, self.loc, self.scale)

        else:
            log_px = log_normal(x, self.loc.clone().detach(), self.scale.clone().detach())

        return log_px

    def as_env(self, act):
        act = act.cpu().detach().numpy()
        env_act = np.tanh(act)
        return env_act


# def inverse_tanh(y):
#
#
#     return 0.5 * (y.log1p() - (-y).log1p())


def inverse_tanh(x):
    eps = 1e-10

    x1p = (x + 1.).clamp(eps, None)
    mx1p = (-x + 1.).clamp(eps, None)

    return 0.5 * (x1p.log() - mx1p.log())


class TanhGaussian(Policy):

    def __init__(self, param):
        super().__init__(param)
        self.loc, self.scale = self.param.chunk(2, dim=-1)
        self.scale = F.softplus(self.scale)

    def sample(self, det=False):

        if det:
            x = self.loc.tanh()

        else:
            noise = self.scale * torch.randn_like(self.scale)
            x = (self.loc + noise).tanh()

        return x

    def log_prob(self, x, param_grad=True):

        z = inverse_tanh(x)

        if param_grad:
            log_pz = log_normal(z, self.loc, self.scale)

        else:
            log_pz = log_normal(z, self.loc.clone().detach(), self.scale.clone().detach())

        #  https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
        # log |d tanh(z) / dz| = log |1 - tanh(z)^2| =

        log_jac = 2. * (math.log(2.) - z - F.softplus(-2. * z))
        log_px = log_pz - log_jac
        return log_px

    def as_env(self, act):
        env_act = act.cpu().detach().numpy()
        return env_act


def cat_sample(p, onehot=False):
    x = torch.multinomial(p.flatten(0, -2), 1, replacement=True)
    x = x.squeeze(-1)
    x = x.unflatten(0, p.shape[: -1])

    if onehot:
        x = F.one_hot(x, p.shape[-1])

    return x


def cat_max(p, onehot=False):
    x = torch.argmax(p, dim=-1)
    if onehot:
        x = F.one_hot(x, p.shape[-1])

    return x


class Cat(Policy):
    def __init__(self, param):
        super().__init__(param)
        self.logits = self.param.log_softmax(dim=-1)
        self.probs = self.logits.exp()

    def log_prob(self, x, param_grad=True):
        if param_grad:
            return (x * self.logits).sum(dim=-1, keepdim=True)
        else:
            return (x * self.logits.clone().detach()).sum(dim=-1, keepdim=True)

    def sample(self, det=False):
        if det:
            return cat_max(self.probs, onehot=True)

        else:
            return cat_sample(self.probs, onehot=True)

    def as_env(self, act):
        act = torch.argmax(act, dim=-1)
        env_act = act.cpu().detach().numpy()
        return env_act

import torch
import numpy as np
import math
import torch.nn.functional as F


def log_normal(x, loc, scale):
    return - scale.log() - 0.5 * (np.log(2. * math.pi) + (x - loc).pow(2.) / scale.pow(2.))


def kld_normal(loc1, s1, loc2, s2):  # KL(p1||p2)

    kld = s2.log() - s1.log() + ((s1.pow(2.) + (loc1 - loc2).pow(2.)) / (s2.pow(2.).mul(2.))) - 0.5
    return kld.sum(dim=-1, keepdim=True)


class Distribution(object):
    def __init__(self, param):
        self.param = param

    def log_prob(self, x, param_grad=True):
        raise NotImplementedError

    def sample(self, det=False):
        raise NotImplementedError


class Gaussian(Distribution):
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

        return log_px.sum(dim=-1, keepdim=True)

    @staticmethod
    def as_env(act):
        act = act.cpu().detach().numpy()
        env_act = np.tanh(act)
        return env_act


def inverse_tanh(x):
    eps = 1e-5
    x1p = (x + 1.).clamp(eps, None)
    mx1p = (-x + 1.).clamp(eps, None)
    z = 0.5 * (x1p.log() - mx1p.log())
    return z


class TanhGaussian(Distribution):

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
        return log_px.sum(dim=-1, keepdim=True)

    @staticmethod
    def as_env(act):
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


class Cat(Distribution):
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
            x = cat_max(self.probs, onehot=True)

        else:
            x = cat_sample(self.probs, onehot=True)

        x = self.probs + (x - self.probs).detach()
        return x

    @staticmethod
    def as_env(act):
        act = torch.argmax(act, dim=-1)
        env_act = act.cpu().detach().numpy()
        return env_act


class MultiCat(Distribution):
    def __init__(self, num_partition, param):
        super().__init__(param)
        self.batch_size = param.size()[0]
        self.num_partition = num_partition

        self.cat_dim = param.size()[1] // self.num_partition

        shaped_params = self.param.unflatten(-1, (self.num_partition, self.cat_dim))

        self.logits = shaped_params.log_softmax(dim=-1).flatten(1, 2)
        self.probs = self.logits.exp()

    def log_prob(self, x, param_grad=True):
        if param_grad:
            return (x * self.logits).sum(dim=-1, keepdim=True)

        else:
            return (x * self.logits.clone().detach()).sum(dim=-1, keepdim=True)

    def sample(self, det=False):

        # [B, partition * code_len] -> [B * partition, code_len]
        probs = self.probs.unflatten(-1, (self.num_partition, self.cat_dim)).flatten(0, 1)

        if det:
            x = cat_max(probs, onehot=True)

        else:
            x = cat_sample(probs, onehot=True)

        x = probs + (x - probs).detach()

        # [B * partition, code_len] -> [B, partition * code_len]
        x = x.unflatten(0, (self.batch_size, self.num_partition)).flatten(1, 2)

        return x

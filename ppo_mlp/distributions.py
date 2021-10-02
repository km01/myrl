import torch
import numpy as np
import torch.nn.functional as F
import math
import gym


def log_normal(x, loc, scale):
    return - scale.log() - 0.5 * (np.log(2. * np.pi) + ((x - loc) / scale).pow(2.))


class TanhGaussian(object):  # not good at A2C/PPO
    def __init__(self, param):
        self.param = param
        self.loc, self.scale = param.chunk(2, dim=-1)
        self.scale = F.softplus(self.scale)

    def sample(self, sample=True):

        if sample:
            noise = self.scale * torch.randn_like(self.scale)
            y = (self.loc + noise).tanh()

        else:
            y = self.loc.tanh()
        return y

    @staticmethod
    def inverse_tanh(y):
        return 0.5 * (y.log1p() - (-y).log1p())

    def log_prob(self, y, param_grad=True):
        y = y.clamp(-0.99999, 0.99999)
        x = self.inverse_tanh(y)

        if param_grad:
            log_px = log_normal(x, self.loc, self.scale)

        else:
            log_px = log_normal(x, self.loc.clone().detach(), self.scale.clone().detach())

        log_jac = 2. * (math.log(2.) - x - F.softplus(-2. * x))
        log_py = log_px - log_jac

        return log_py

    @staticmethod
    def env_action(act):
        action = act.cpu().detach().numpy()
        return action


class Gaussian(object):
    def __init__(self, param):
        self.param = param
        self.loc, self.scale = param.chunk(2, dim=-1)
        self.scale = F.softplus(self.scale)

    def sample(self, sample=True):

        if sample:
            noise = self.scale * torch.randn_like(self.scale)
            x = self.loc + noise

        else:
            x = self.loc
        return x

    def log_prob(self, x, param_grad=True):

        if param_grad:
            log_px = log_normal(x, self.loc, self.scale)

        else:
            log_px = log_normal(x, self.loc.clone().detach(), self.scale.clone().detach())

        return log_px

    @staticmethod
    def env_action(act):
        act = act.cpu().detach().numpy()
        action = np.tanh(act)
        return action


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


class Cat(object):
    def __init__(self, param):
        self.param = param
        self.logits = param.log_softmax(dim=-1)
        self.probs = self.logits.exp()

    def log_prob(self, x, param_grad=True):
        if param_grad:
            return (x * self.logits).sum(dim=-1, keepdim=True)
        else:
            return (x * self.logits.clone().detach()).sum(dim=-1, keepdim=True)

    def sample(self, sample=True):
        if sample:
            return cat_sample(self.probs, onehot=True)

        else:
            return cat_max(self.probs, onehot=True)

    @staticmethod
    def env_action(act):
        action = torch.argmax(act, dim=-1)
        action = action.cpu().detach().numpy()
        return action


def get_proper_policy_class(env):
    if type(env.action_space) == gym.spaces.box.Box:
        policy_class = Gaussian
        num_params = env.action_space.shape[0] * 2

    elif type(env.action_space) == gym.spaces.discrete.Discrete:
        policy_class = Cat
        num_params = env.action_space.n

    else:
        raise NotImplementedError

    return policy_class, num_params

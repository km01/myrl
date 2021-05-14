import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


_stddev_EPS = 1e-8
_theta_EPS = 1e-8


class NormalPredictor(nn.Module):
    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_size, h_size),
                                 nn.ELU(),
                                 nn.Linear(h_size, z_size + z_size))

    def forward(self, x):
        x = self.net(x)
        x = self.to_param(x)
        return x

    @staticmethod
    def to_param(x):
        loc, scale = x.chunk(2, dim=-1)
        scale = F.softplus(scale) + _stddev_EPS
        return [loc, scale]

    @staticmethod
    def sample(loc, scale):
        return loc + torch.randn_like(scale) * scale

    @staticmethod
    def log_prob(x, loc, scale):
        c = np.log(2. * np.pi)
        log_px = - scale.log() - 0.5 * (c + ((x - loc) / scale).pow(2.))
        return log_px


class BernoulliPredictor(nn.Module):
    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_size, h_size),
                                 nn.ELU(),
                                 nn.Linear(h_size, z_size + z_size))

    def forward(self, x):
        x = self.net(x)
        x = self.to_param(x)
        return x

    @staticmethod
    def to_param(x):
        theta = x.sigmoid()
        return [theta]

    @staticmethod
    def sample(theta):
        x = torch.bernoulli(theta)
        # x = theta - (theta - x).detach().clone()
        return x

    @staticmethod
    def log_prob(x, theta):
        log_px = x * (theta + _theta_EPS).log() + (1. - x) * (1. - theta + _theta_EPS).log()
        return log_px


class TanhGaussian(object):
    def __init__(self, base_loc, base_scale):
        self.x_loc = base_loc
        self.x_scale = base_scale

    def sample(self, mode=False):
        if mode:
            noise = 0.
        else:
            noise = self.x_scale * torch.randn_like(self.x_scale)

        y = (self.x_loc + noise).tanh()
        return y

    @staticmethod
    def inverse_tanh(y):
        x = 0.5 * ((y + 1. + 1e-8).log() - (1. - y + 1e-8).log())
        return x

    def log_normal(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc
        std = self.x_scale
        log_px = - std.log() - 0.5 * (c + ((x - mean)/std).pow(2.))
        return log_px

    def log_prob(self, y):
        x = self.inverse_tanh(y)
        log_px = self.log_normal(x)
        jac = (1. - x.tanh().pow(2.)).abs() + 1e-8
        log_py = log_px - jac.log()
        return log_py

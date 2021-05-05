import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time


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
        x = 0.5 * ((y + 1 + 1e-8).log() - (1. - y + 1e-8).log())
        return x

    def log_normal(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc
        std = self.x_scale + 1e-8
        log_px = - std.log() - 0.5 * (c + ((x - mean)/std).pow(2.))
        return log_px

    def log_prob(self, y):
        x = self.inverse_tanh(y)
        log_px = self.log_normal(x)
        jac = (1. - x.tanh().pow(2.)).abs() + 1e-8
        log_py = log_px - jac.log()
        return log_py


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.min_action = -2.
        self.max_action = 2.
        self.dist = TanhGaussian

    def forward(self, x):
        loc, scale = self.net(x).chunk(2, dim=-1)
        loc = loc.tanh()
        scale = F.softplus(scale)
        return [loc, scale]

    def transform_action(self, act):
        act = (self.max_action - self.min_action) * 0.5 * act
        act = act + 0.5 * (self.max_action + self.min_action)
        return act


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, a):
        x_and_a = torch.cat([x, a], dim=-1)
        q_val = self.net(x_and_a)
        return q_val


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        val = self.net(x)
        return val


@torch.no_grad()
def render_simulation(env_name, actor):
    env = gym.make(env_name)
    with torch.no_grad():
        obs = env.reset()
        while True:
            env.render()
            time.sleep(0.01)
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = actor.dist(*actor(x)).sample(mode=True)
            real_action = actor.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            if done:
                break
    env.close()


@torch.no_grad()
def test_env(env_name, actor):
    env = gym.make(env_name)
    reward_sum = 0.0
    with torch.no_grad():
        obs = env.reset()
        while True:
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = actor.dist(*actor(x)).sample(mode=True)
            real_action = actor.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            reward_sum += rew
            if done:
                break
    env.close()
    return reward_sum


def transform_data(batch):
    obs, act, rew, done, obs_next = batch
    obs = torch.FloatTensor(obs)
    act = torch.FloatTensor(act)
    rew = torch.FloatTensor(rew).unsqueeze(-1)
    rew = (rew + 8.) / 8.
    done = torch.BoolTensor(done).unsqueeze(-1)
    obs_next = torch.FloatTensor(obs_next)
    return [obs, act, rew, done, obs_next]


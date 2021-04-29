import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.min_action = -2.
        self.max_action = 2.
        self.noise_std = 0.1
        self.noise_clip = 0.5

    def forward(self, x):
        act = self.net(x)
        return act.tanh()

    def add_noise(self, act):  # exploration and target policy smoothing
        noise = torch.randn_like(act)
        noise.clamp_(-self.noise_clip, self.noise_clip)
        act += self.noise_std * noise
        act.clamp_(-1., 1.)
        return act

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

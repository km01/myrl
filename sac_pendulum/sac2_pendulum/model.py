import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        loc, scale = self.net(x).chunk(2, dim=-1)
        scale = F.softplus(scale)
        return [loc, scale]


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        q1, q2 = self.q1(xa), self.q2(xa)
        return q1, q2

    def evaluate(self, x, a):
        q1, q2 = self(x, a)
        return torch.min(q1, q2)

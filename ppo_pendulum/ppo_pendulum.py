from ppo import PPO
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus


class PendulumPPO(PPO):
    def __init__(self):
        super().__init__(distribution_type=Normal)
        self.a_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.c_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        params = self.policy(x)
        value = self.critic(x)
        return params, value

    def critic(self, x):
        return self.c_net(x)

    def policy(self, x):
        loc, scale = self.a_net(x).chunk(2, dim=-1)
        scale = softplus(scale)
        return [loc, scale]

    def response(self, x, mode=False):
        loc, scale = self.policy(x)
        if mode:
            act = loc

        else:
            act = Normal(loc, scale).sample()

        return act, [loc, scale]

    def to_real_action(self, act):
        return act.tanh().mul(2.)


class PendulumPPO2(PPO):
    def __init__(self):
        super().__init__(distribution_type=Normal)
        self.a_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.c_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        params = self.policy(x)
        value = self.critic(x)
        return params, value

    def critic(self, x):
        return self.c_net(x)

    def policy(self, x):
        loc, scale = self.a_net(x).chunk(2, dim=-1)
        loc = loc.tanh().mul(2.)
        scale = softplus(scale)
        return [loc, scale]

    def response(self, x, mode=False):
        loc, scale = self.policy(x)
        if mode:
            act = loc

        else:
            act = Normal(loc, scale).sample()

        act = act.clamp(-2, 2)
        return act, [loc, scale]

    def to_real_action(self, act):
        return act

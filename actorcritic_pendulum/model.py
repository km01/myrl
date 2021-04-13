import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        mu, sig = self.net(x).chunk(2, dim=-1)
        mu = 2.0 * mu.tanh()
        sig = sig.exp()
        return mu, sig


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def compute_td(next_value, rewards, is_terminals, gamma=0.99):
    """
    :param next_value: [V(s_{t+n})]
    :param rewards: [r_{t}, r_{t+1}, r_{t+2}, ... , r_{t+n-1}]
    :param is_terminals: [t_{t+1}, t_{t+2}, t_{t+3}, ... , t_{t+n}]
    :param gamma:
    :return: [1-step td, 2-step td, ...., n-step td]
    """

    rewards = torch.FloatTensor(rewards).to(next_value.device)
    is_terminals = torch.BoolTensor(is_terminals).to(next_value.device)
    ret = next_value

    trgs = []
    for rew, terminal in zip(reversed(rewards), reversed(is_terminals)):
        ret.masked_fill_(terminal, 0.)
        ret = rew + gamma * ret
        trgs.append(ret)
    trgs.reverse()
    trgs = torch.stack(trgs, dim=0)
    return trgs
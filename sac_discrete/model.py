import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def create_mlp(dimension_list, output_module=None):

    d, num_layers, mlp = dimension_list, len(dimension_list) - 1, []
    assert num_layers > 0, 'invalid dimension_list'

    for i in range(num_layers):
        mlp += [nn.Linear(d[i], d[i + 1])]
        if i < num_layers - 1:
            mlp += [nn.ELU()]

        elif output_module is not None:
            mlp += [output_module]

    mlp = nn.Sequential(*mlp)
    return mlp


class MLP(nn.Module):
    def __init__(self, dimension_list, output_module=None):
        super().__init__()
        self.net = create_mlp(dimension_list, output_module)
        self.apply(weights_init_)

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = MLP([input_size, hidden_size, hidden_size, 1])

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        d = [input_size + action_size, hidden_size, hidden_size, 1]
        self.net1, self.net2 = MLP(d), MLP(d)

    def forward(self, x, a):
        inp = torch.cat([x, a], dim=-1)
        q1, q2 = self.net1(inp), self.net2(inp)
        return q1, q2


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        d = [input_size, hidden_size, hidden_size, action_size]
        self.net1, self.net2 = MLP(d), MLP(d)

    def forward(self, x):
        q1, q2 = self.net1(x), self.net2(x)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.net = MLP([input_size, hidden_size, hidden_size, action_size])

    def forward(self, x):
        x = self.net(x)
        return x

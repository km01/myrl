import torch
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def to_mlp_dim(input_size, hidden_size, num_hidden_layers, output_size):
    return [input_size] + [hidden_size] * num_hidden_layers + [output_size]


def create_mlp(dimension_list, output_module=None):
    d, num_layers, mlp = dimension_list, len(dimension_list) - 1, []
    assert num_layers > 0, 'invalid dimension_list'

    for i in range(num_layers):
        mlp += [nn.Linear(d[i], d[i + 1])]
        if i < num_layers - 1:
            mlp += [nn.ReLU()]

        elif output_module is not None:
            mlp += [output_module]

    mlp = nn.Sequential(*mlp)
    return mlp


class MLP(nn.Module):
    def __init__(self, dim_list, output_module=None):
        super().__init__()
        self.net = create_mlp(dim_list, output_module)
        self.apply(weights_init_)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self,
                 input_size,
                 action_param_size,
                 hidden_size,
                 num_shared_layers,
                 num_actor_layers,
                 num_critic_layers,
                 policy_distribution,
                 device):
        super().__init__()

        base_dim_list = [input_size] + [hidden_size] * num_shared_layers
        actor_dim_list = [hidden_size] * num_actor_layers + [action_param_size]
        critic_dim_list = [hidden_size] * num_critic_layers + [1]
        self.base = MLP(base_dim_list, nn.ReLU()).to(device)
        self.actor = MLP(actor_dim_list).to(device)
        self.critic = MLP(critic_dim_list).to(device)
        self.policy_distribution = policy_distribution
        self.device = device

    def forward(self, x):
        x = self.base(x)
        policy = self.policy_distribution(self.actor(x))
        val_pred = self.critic(x)
        return policy, val_pred

    @torch.no_grad()
    def select_action(self, x, sample=True):
        x = torch.FloatTensor(x).to(self.device)
        policy, val = self.forward(x)
        act = policy.sample(sample)
        log_prob = policy.log_prob(act)
        env_action = policy.env_action(act)

        agent_info = {'act': act.cpu().detach().numpy(),
                      'log_prob': log_prob.cpu().detach().numpy(),
                      'val': val.cpu().detach().numpy()}

        return env_action, agent_info


@torch.no_grad()
def compute_ret(vals, rews, dones, truncated, gamma=0.99, lamda=0.95, next_val=None):
    roll_len = vals.size(0)

    vals = vals.squeeze(-1)
    rets = torch.zeros_like(vals)

    if next_val is None:
        next_v = torch.zeros_like(vals[0])
        truncated = truncated.clone().detach()
        truncated[-1, :] = True

    else:
        next_v = next_val.squeeze(-1)

    if lamda is not None:   # gae
        gae = torch.zeros_like(next_v)
        for t in reversed(range(roll_len)):
            val, rew, done, trunc = vals[t], rews[t], dones[t], truncated[t]
            next_v.masked_fill_(done, 0.)
            gae.masked_fill_(done, 0.)
            delta = rew - val + gamma * next_v
            gae = delta + gamma * lamda * gae
            gae.masked_fill_(trunc, 0.)
            rets[t].copy_(gae + val)
            next_v.copy_(val)

    else:   # naive mc
        ret = next_v
        for t in reversed(range(roll_len)):
            val, rew, done, trunc = vals[t], rews[t], dones[t], truncated[t]
            ret.masked_fill_(done, 0.)
            ret = rew + gamma * ret
            ret.masked_fill_(trunc, 0.)
            ret += val.masked_fill(torch.logical_not(trunc), 0.)
            rets.copy_(ret)

    rets = rets.unsqueeze(-1)
    return rets

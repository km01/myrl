import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
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
        mu, logvar = self.net(x).chunk(2, dim=-1)
        mu = mu.tanh().mul(2.)
        std = logvar.mul(0.5).exp()
        return mu, std


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def sample_n_step(ob_env_now, actor, envs, n):
    cache = []
    for _ in range(n):
        ob_env = envs.reset() if ob_env_now is None else ob_env_now
        x = torch.FloatTensor(ob_env)
        loc, scale = actor(x)
        act = Normal(loc, scale).sample().clamp(-2.0, 2.0).numpy()
        ob_env_now, rew, done, _ = envs.step(act)
        cache.append([ob_env, act, rew, done, loc.numpy(), scale.numpy()])

    cache = [np.stack(val) for val in zip(*cache)]
    return ob_env_now, *cache


@torch.no_grad()
def compute_gae(ob_env, rew, done, ob_env_next, critic, gamma=0.95, tau=0.9):
    ob_env = np.append(ob_env, [ob_env_next], axis=0)
    ob_env = torch.FloatTensor(ob_env)
    rew = torch.FloatTensor(rew)
    done = torch.BoolTensor(done)
    values = critic(ob_env).squeeze(-1)
    next_v = values[-1]
    values = values[:-1]
    g_a_e = []
    gae = torch.zeros_like(next_v)
    for r, d, v in reversed(list(zip(rew, done, values))):
        gae = r - v + gamma * (next_v + tau * gae).masked_fill(d, 0.)
        g_a_e.insert(0, gae)
        next_v = v

    g_a_e = torch.stack(g_a_e, dim=0).numpy()
    values = values.numpy()
    return g_a_e, values


def ppo_train(memory, batch_size, ppo_epoch, actor, critic, a_solver, c_solver, clip_param, max_grad_norm):
    action_loss_mean, value_loss_mean = 0., 0.
    for _ in range(ppo_epoch):
        batch = memory.sample(batch_size)
        ob_env, act, old_gae, old_val, old_loc, old_scale = [torch.FloatTensor(x) for x in batch]

        old_gae = old_gae.unsqueeze(-1)
        old_val = old_val.unsqueeze(-1)

        old_dist = Normal(old_loc, old_scale)

        new_loc, new_scale = actor(ob_env)
        new_val = critic(ob_env)
        new_dist = Normal(new_loc, new_scale)

        test_version = True

        if test_version:
            gae = old_gae + old_val - new_val.detach()
            critic_targ = old_gae + new_val.detach()
        else:
            gae = old_gae
            critic_targ = old_gae + old_val

        ratio = torch.exp(new_dist.log_prob(act) - old_dist.log_prob(act))
        surr1 = ratio * gae
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * gae
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.smooth_l1_loss(new_val, critic_targ)

        a_solver.zero_grad()
        c_solver.zero_grad()

        loss = action_loss + value_loss
        loss.backward()

        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        a_solver.step()
        c_solver.step()

        action_loss_mean += action_loss.item() / ppo_epoch
        value_loss_mean += value_loss.item() / ppo_epoch
    return action_loss_mean, value_loss_mean

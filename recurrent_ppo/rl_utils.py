import gym
import numpy as np
from distributions import Cat, Gaussian, TanhGaussian
import torch
from env_utils import make_vec_env


@torch.no_grad()
def compute_ret(vals, rews, dones, truncated, gamma=0.99, lamda=0.95, next_val=None):
    roll_len = vals.size(0)

    vals = vals.squeeze(-1)
    rets = torch.zeros_like(vals)

    if next_val is None:
        next_v = torch.zeros_like(vals[0])
        truncated = truncated.clone().detach()
        truncated[-1] = True

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


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')

        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):

        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def normalize(self, x):
        self.update(x)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def get_proper_policy_class(env, tanh_if_continuous=False):
    if type(env.action_space) == gym.spaces.box.Box:
        policy_class = Gaussian
        if tanh_if_continuous:
            policy_class = TanhGaussian

        num_action = env.action_space.shape[0]
        num_params = num_action * 2

    elif type(env.action_space) == gym.spaces.discrete.Discrete:
        policy_class = Cat
        num_action = env.action_space.n
        num_params = num_action

    else:
        raise NotImplementedError

    return policy_class, num_action, num_params

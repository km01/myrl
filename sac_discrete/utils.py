import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

import torch
import random


utils_epsilon = 1e-8


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def cat_sample(p, onehot=False):
    x = torch.multinomial(p.flatten(0, -2), 1, replacement=True)
    x = x.squeeze(-1)
    x = x.unflatten(0, p.shape[: -1])

    if onehot:
        x = F.one_hot(x, p.shape[-1])

    return x


def cat_max(p, onehot=False):
    x = torch.argmax(p, dim=-1)
    if onehot:
        x = F.one_hot(x, p.shape[-1])

    return x


class Cat(object):
    def __init__(self, probs=None, logits=None, raw_base=None):
        assert int(probs is not None) + int(logits is not None) + int(raw_base is not None) == 1, \
            'Cat gets one parameter'

        self.probs, self.logits, self.raw_base = probs, logits, raw_base
        self.logsumexp = None

        if self.probs is not None:
            self.logits = self.probs.log()

        elif self.logits is not None:
            self.probs = self.logits.exp()

        else:
            self.logsumexp = torch.logsumexp(self.raw_base, dim=-1, keepdim=False)
            self.logits = self.raw_base.log_softmax(dim=-1)
            self.probs = self.logits.exp()

    def log_prob(self, x, param_grad=True):
        if param_grad:
            return (x * self.logits).sum(dim=-1)
        else:
            return (x * self.logits.clone().detach()).sum(dim=-1)

    def sample(self, onehot=False, sample=True):
        if sample:
            return cat_sample(self.probs, onehot)

        else:
            return cat_max(self.probs, onehot)

    def rsample(self, sample=True):
        x = self.sample(onehot=True, sample=sample)
        x = self.probs + (x - self.probs).detach()  # straight through
        return x


def sample_episode(memory, env, num_samples, model=None):

    sample_count = 0
    assert num_samples > 0, '?'
    while True:

        obs, done, life_span = env.reset(), False, 0
        while not done:
            if model is None:
                act = env.action_space.sample()
            else:
                act = model.select_action(obs)

            obs_next, rew, done, _ = env.step(act)
            life_span += 1
            sample_count += 1
            real_done = False if life_span == env._max_episode_steps else done
            memory.push([obs, act, rew, real_done, obs_next])

            if sample_count == num_samples:
                env.reset()
                return

            obs = obs_next


def evaluate_episode(model, env, num_test_episode):

    gain_avg, life_span_avg = 0, 0
    for _ in range(num_test_episode):
        gain, life_span = 0, 0
        obs, done = env.reset(), False
        while not done:

            act = model.select_action(obs, sample=False)
            obs_next, rew, done, _ = env.step(act)
            life_span += 1
            gain += rew
            obs = obs_next
        gain_avg += gain
        life_span_avg += life_span

    gain_avg = gain_avg / num_test_episode
    life_span_avg = life_span_avg / num_test_episode
    return gain_avg, life_span_avg


def run_episode(model, env, memory, batch_size, updates_per_step):

    assert len(memory) >= batch_size, 'num_initial_steps should be larger than batch_size'

    gain, life_span = 0, 0
    obs, done = env.reset(), False
    loss_stats = None
    while not done:

        act = model.select_action(obs)
        for i in range(updates_per_step):
            loss_info = model.update_parameters(memory.sample(batch_size))

            if loss_stats is None:
                loss_stats = loss_info

            else:
                for key in loss_stats:
                    loss_stats[key] += loss_info[key]

        obs_next, rew, done, _ = env.step(act)
        life_span += 1
        gain += rew
        real_done = False if life_span == env._max_episode_steps else done
        memory.push([obs, act, rew, real_done, obs_next])
        obs = obs_next

    for key in loss_stats:
        loss_stats[key] = round(loss_stats[key] / updates_per_step, 2)

    return gain, life_span, loss_stats

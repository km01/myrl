import numpy as np
import torch
import torch.nn.functional as F
import random

utils_epsilon = 1e-8


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class TanhGaussian(object):
    def __init__(self, base_loc, base_scale):
        self.x_loc = base_loc
        self.x_scale = base_scale

    def sample(self, sample=True):

        if sample:
            noise = self.x_scale * torch.randn_like(self.x_scale)
            y = (self.x_loc + noise).tanh()

        else:
            y = self.x_loc.tanh()
        return y

    @staticmethod
    def inverse_tanh(y):
        x = 0.5 * ((y + 1. + 1e-8).log() - (1. - y + 1e-8).log())
        x = 0.5 * (y.log1p() - (-y).log1p())
        return x

    def log_normal(self, x):
        c = np.log(2. * np.pi)
        log_px = - (self.x_scale + 1e-9).log() - 0.5 * (c + ((x - self.x_loc) / (self.x_scale + 1e-9)).pow(2.))
        return log_px

    def log_normal_no_grad_params(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc.detach().clone()
        std = self.x_scale.detach().clone() + 1e-9
        log_px = - std.log() - 0.5 * (c + ((x - mean) / std).pow(2.))
        return log_px

    def log_prob(self, y, param_grad=False):
        y = y.clamp(-0.9999, 0.9999)
        x = self.inverse_tanh(y)

        if param_grad:
            log_px = self.log_normal(x)
            log_jac = 0. * (np.log(2.) - x - F.softplus(-2. * x))
            log_py = log_px - log_jac

        else:
            log_px = self.log_normal_no_grad_params(x)
            log_jac = 2. * (np.log(2.) - x - F.softplus(-2. * x))
            log_py = log_px - log_jac

        return log_py


class Gaussian(object):
    def __init__(self, base_loc, base_scale):
        self.x_loc = base_loc
        self.x_scale = base_scale

    def sample(self, sample=True):

        if sample:
            noise = self.x_scale * torch.randn_like(self.x_scale)
            y = self.x_loc + noise

        else:
            y = self.x_loc
        return y

    def log_normal(self, x):
        c = np.log(2. * np.pi)
        log_px = - (self.x_scale + 1e-8).log() - 0.5 * (c + ((x - self.x_loc) / (self.x_scale + 1e-8)).pow(2.))
        return log_px

    def log_normal_no_grad_params(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc.detach().clone()
        std = self.x_scale.detach().clone() + 1e-8
        log_px = - std.log() - 0.5 * (c + ((x - mean) / std).pow(2.))
        return log_px

    def log_prob(self, x, param_grad=False):

        if param_grad:
            log_px = self.log_normal(x)

        else:
            log_px = self.log_normal_no_grad_params(x)

        return log_px


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ActionTransform(object):
    def __init__(self, low, high):
        self.low, self.high = low, high

    def __call__(self, action):
        action = (self.high - self.low) * 0.5 * action
        action = action + 0.5 * (self.high + self.low)
        return action

    def inv(self, y):
        y = y - 0.5 * (self.high + self.low)
        y = 2 * y / (self.high - self.low)
        return y


def run_episode(model, env, memory, batch_size, updates_per_step):

    assert len(memory) >= batch_size, 'num_initial_steps should be larger than batch_size'

    gain, life_span = 0, 0
    obs, done = env.reset(), False
    tr = ActionTransform(env.action_space.low, env.action_space.high)
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

        obs_next, rew, done, _ = env.step(tr(act))
        life_span += 1
        gain += rew
        done_ = False if life_span == env._max_episode_steps else done
        memory.push([obs, act, rew, done_, obs_next])
        obs = obs_next

    for key in loss_stats:
        loss_stats[key] = round(loss_stats[key] / updates_per_step, 2)

    return gain, life_span, loss_stats


def sample_episode(memory, env, num_samples):

    tr = ActionTransform(env.action_space.low, env.action_space.high)

    sample_count = 0
    assert num_samples > 0, '?'
    while True:

        obs, done, life_span = env.reset(), False, 0
        while not done:
            act = env.action_space.sample()
            obs_next, rew, done, _ = env.step(tr(act))
            life_span += 1
            sample_count += 1
            done_ = False if life_span == env._max_episode_steps else done
            memory.push([obs, act, rew, done_, obs_next])

            if sample_count == num_samples:
                env.reset()
                return

            obs = obs_next


def evaluate_episode(model, env, num_test_episode):

    tr = ActionTransform(env.action_space.low, env.action_space.high)

    gain_avg, life_span_avg = 0, 0
    for _ in range(num_test_episode):
        gain, life_span = 0, 0
        obs, done = env.reset(), False
        while not done:

            act = model.select_action(obs, sample=False)
            obs_next, rew, done, _ = env.step(tr(act))
            life_span += 1
            gain += rew
            obs = obs_next
        gain_avg += gain
        life_span_avg += life_span

    gain_avg = gain_avg / num_test_episode
    life_span_avg = life_span_avg / num_test_episode
    return gain_avg, life_span_avg


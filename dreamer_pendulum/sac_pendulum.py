import torch
import torch.nn as nn
import gym
import time
from utils import NormalPredictor, TanhGaussian


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.min_action = -2.
        self.max_action = 2.
        self.dist = TanhGaussian

    def forward(self, x):
        return NormalPredictor.to_param(self.net(x))

    def transform_action(self, act):
        act = (self.max_action - self.min_action) * 0.5 * act
        act = act + 0.5 * (self.max_action + self.min_action)
        return act


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
        x_and_a = torch.cat([x, a], dim=-1)
        q1 = self.q1(x_and_a)
        return q1

    def loss_fn(self, x, a, q_targ):
        x_and_a = torch.cat([x, a], dim=-1)
        q1 = self.q1(x_and_a)
        q2 = self.q2(x_and_a)
        loss = (q1 - q_targ).pow(2.) + (q2 - q_targ).pow(2.)
        loss = loss.mul(0.5)
        return loss


class Critic2(nn.Module):
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
        x_and_a = torch.cat([x, a], dim=-1)
        q1 = self.q1(x_and_a)
        q2 = self.q2(x_and_a)
        q, _ = torch.stack([q1, q2], dim=-1).min(dim=-1, keepdim=False)
        return q

    def loss_fn(self, x, a, q_targ):
        x_and_a = torch.cat([x, a], dim=-1)
        q1 = self.q1(x_and_a)
        q2 = self.q2(x_and_a)
        loss = (q1 - q_targ).pow(2.) + (q2 - q_targ).pow(2.)
        loss = loss.mul(0.5)
        return loss


@torch.no_grad()
def render_simulation(env_name, actor):
    env = gym.make(env_name)
    with torch.no_grad():
        obs = env.reset()
        while True:
            env.render()
            time.sleep(0.01)
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = actor.dist(*actor(x)).sample(mode=True)
            real_action = actor.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            if done:
                break
    env.close()


@torch.no_grad()
def test_env(env_name, actor):
    env = gym.make(env_name)
    reward_sum = 0.0
    with torch.no_grad():
        obs = env.reset()
        while True:
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = actor.dist(*actor(x)).sample(mode=True)
            real_action = actor.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            reward_sum += rew
            if done:
                break
    env.close()
    return reward_sum


def transform_data(batch):
    obs, act, rew, done, obs_next = batch
    obs = torch.FloatTensor(obs)
    act = torch.FloatTensor(act)
    rew = torch.FloatTensor(rew).unsqueeze(-1)
    rew = (rew + 8.) / 8.
    done = torch.BoolTensor(done).unsqueeze(-1)
    obs_next = torch.FloatTensor(obs_next)
    return [obs, act, rew, done, obs_next]

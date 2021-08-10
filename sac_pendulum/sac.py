import copy
import numpy as np
import torch


class TanhGaussian(object):
    def __init__(self, base_loc, base_scale):
        self.x_loc = base_loc
        self.x_scale = base_scale

    def sample(self, mode=False):

        if mode:
            noise = 0.
        else:
            noise = self.x_scale * torch.randn_like(self.x_scale)

        y = (self.x_loc + noise).tanh()
        return y

    @staticmethod
    def inverse_tanh(y):
        x = 0.5 * ((y + 1. + 1e-8).log() - (1. - y + 1e-8).log())
        return x

    def log_normal(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc
        std = self.x_scale + 1e-8
        log_px = - std.log() - 0.5 * (c + ((x - mean) / std).pow(2.))
        return log_px

    def log_normal_no_grad_params(self, x):
        c = np.log(2. * np.pi)
        mean = self.x_loc.detach().clone()
        std = self.x_scale.detach().clone() + 1e-8
        log_px = - std.log() - 0.5 * (c + ((x - mean) / std).pow(2.))
        return log_px

    def log_prob(self, y, no_grad_params=False):
        x = self.inverse_tanh(y)
        if no_grad_params:
            log_px = self.log_normal_no_grad_params(x)
        else:
            log_px = self.log_normal(x)
        jac = (1. - x.tanh().pow(2.)).abs() + 1e-8
        log_py = log_px - jac.log()
        return log_py


class SAC(object):
    def __init__(self,
                 actor,
                 critic,
                 gamma=0.99,
                 alpha=0.001,
                 soft_tau=0.005,
                 actor_lr=3e-4,
                 critic_lr=3e-4):

        self.actor = actor
        self.critic = critic
        self.alpha = alpha

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

        self.gamma = gamma
        self.soft_tau = soft_tau

        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, obs, mode=False):
        obs = torch.FloatTensor(obs)

        if mode:
            act = self.actor(obs)[0].cpu().numpy()
        else:
            act = TanhGaussian(*self.actor(obs)).sample().cpu().numpy()
        return act

    def train(self, batch):
        obs, act, rew, done, obs_next = batch
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew).unsqueeze(-1)
        done = torch.BoolTensor(done).unsqueeze(-1)
        obs_next = torch.FloatTensor(obs_next)

        with torch.no_grad():
            target_policy = TanhGaussian(*self.actor(obs_next))
            target_action = target_policy.sample()
            next_q = self.critic_target.evaluate(obs_next, target_action)
            next_v = next_q - self.alpha * target_policy.log_prob(target_action)
            q_targ = rew + self.gamma * next_v.masked_fill(done, 0.)

        self.c_optim.zero_grad()
        q1, q2 = self.critic(obs, act)
        critic_loss = (q1 - q_targ).pow(2.) + (q2 - q_targ).pow(2.)
        critic_loss = critic_loss.mean().mul(0.5)
        critic_loss.backward()
        self.c_optim.step()

        self.a_optim.zero_grad()
        policy = TanhGaussian(*self.actor(obs))
        action = policy.sample()
        q_pred = self.critic.evaluate(obs, action)
        actor_loss = self.alpha * policy.log_prob(action, True) - q_pred
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.a_optim.step()

        self.update_target()

    def update_target(self):
        for w, w_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            w_target.data.copy_(self.soft_tau * w.data + (1 - self.soft_tau) * w_target.data)

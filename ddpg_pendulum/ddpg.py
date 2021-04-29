import copy
import numpy as np
import torch


class DDPG(object):
    def __init__(self,
                 actor,
                 critic,
                 gamma=0.99,
                 soft_tau=0.005,
                 actor_lr=3e-4,
                 critic_lr=3e-4):

        self.actor = actor
        self.critic = critic

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.requires_grad_(False)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

        self.gamma = gamma
        self.soft_tau = soft_tau

        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def __call__(self, x):
        return self.actor(x)

    def add_noise(self, act):
        return self.actor.add_noise(act)

    def transform_action(self, act):
        return self.actor.transform_action(act)

    def train(self, batch, tps=True):
        obs, act, rew, done, obs_next = batch

        # compute TD target
        next_act_target = self.actor_target(obs_next)
        with torch.no_grad():
            if tps:  # target policy smoothing.. introduced by TD3 paper
                next_act_target = self.add_noise(next_act_target)

            td_q = self.critic_target(obs_next, next_act_target)
            td_q = rew + self.gamma * td_q.masked_fill(done, 0.)

        self.c_optim.zero_grad()
        q_pred = self.critic(obs, act)
        critic_loss = (q_pred - td_q).pow(2.).mean().mul(0.5)
        critic_loss.backward()
        self.c_optim.step()

        self.a_optim.zero_grad()
        q_pred = self.critic(obs, self.actor(obs))
        actor_loss = q_pred.mean().mul(-1)
        actor_loss.backward()
        self.a_optim.step()
        self.update_target()

    def update_target(self):
        for w, w_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            w_target.data.copy_(self.soft_tau * w.data + (1 - self.soft_tau) * w_target.data)

        for w, w_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            w_target.data.copy_(self.soft_tau * w.data + (1 - self.soft_tau) * w_target.data)


@torch.no_grad()
def n_step(obs_next, envs, ddpg, n):
    trans = []
    for _ in range(n):
        obs = envs.reset() if obs_next is None else obs_next
        x = torch.FloatTensor(obs)
        act = ddpg(x)
        act = ddpg.add_noise(act)
        real_action = ddpg.transform_action(act)
        obs_next, rew, done, _ = envs.step(real_action.numpy())
        trans.append([obs, act, rew, done, obs_next])

    trans = [np.stack(val) for val in zip(*trans)]
    return obs_next, trans


def transform_data(batch):
    obs, act, rew, done, obs_next = batch
    obs = torch.FloatTensor(obs)
    act = torch.FloatTensor(act)
    rew = torch.FloatTensor(rew).unsqueeze(-1)
    rew = (rew + 8.) / 8.
    done = torch.BoolTensor(done).unsqueeze(-1)
    obs_next = torch.FloatTensor(obs_next)
    return [obs, act, rew, done, obs_next]

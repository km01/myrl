import copy
import numpy as np
import torch


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

    def train(self, batch):

        dist = self.actor.dist
        obs, old_act, rew, done, obs_next = batch

        with torch.no_grad():
            targ_policy = dist(*self.actor(obs_next))
            targ_act = targ_policy.sample()
            next_q = self.critic_target(obs_next, targ_act)
            next_v = next_q - self.alpha * targ_policy.log_prob(targ_act)
            q_targ = rew + self.gamma * next_v.masked_fill(done, 0.)

        self.c_optim.zero_grad()
        critic_loss = self.critic.loss_fn(obs, old_act, q_targ)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        self.c_optim.step()

        policy = dist(*self.actor(obs))
        new_act = policy.sample()
        q_now = self.critic(obs, new_act)

        self.a_optim.zero_grad()
        actor_loss = self.alpha * policy.log_prob(new_act, no_grad_params=True) - q_now
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.a_optim.step()

        self.update_target()

    def update_target(self):
        for w, w_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            w_target.data.copy_(self.soft_tau * w.data + (1 - self.soft_tau) * w_target.data)


@torch.no_grad()
def n_step(obs_next, envs, actor, n):
    trans = []
    for _ in range(n):
        obs = envs.reset() if obs_next is None else obs_next
        x = torch.FloatTensor(obs)
        act = actor.dist(*actor(x)).sample()
        real_action = actor.transform_action(act)
        obs_next, rew, done, _ = envs.step(real_action.numpy())
        trans.append([obs, act, rew, done, obs_next])

    trans = [np.stack(val) for val in zip(*trans)]
    return obs_next, trans

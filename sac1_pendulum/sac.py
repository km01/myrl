import copy
import numpy as np
import torch


class SAC(object):
    def __init__(self,
                 actor,
                 critic,
                 value_net,
                 gamma=0.99,
                 alpha=0.001,
                 soft_tau=0.005,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 value_net_lr=3e-4):

        self.actor = actor
        self.critic = critic
        self.alpha = alpha
        self.value_net = value_net
        self.value_net_target = copy.deepcopy(self.value_net)
        self.value_net_target.requires_grad_(False)

        self.gamma = gamma
        self.soft_tau = soft_tau

        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.v_optim = torch.optim.Adam(self.value_net.parameters(), lr=value_net_lr)

    def train(self, batch):
        obs, old_act, rew, done, obs_next = batch

        with torch.no_grad():
            next_v = self.value_net_target(obs_next)
            q_targ = rew + self.gamma * next_v.masked_fill(done, 0.)

        self.c_optim.zero_grad()
        q_eval = self.critic(obs, old_act)
        critic_loss = (q_eval - q_targ).pow(2.).mean().mul(0.5)
        critic_loss.backward()
        self.c_optim.step()

        params = self.actor(obs)
        policy = self.actor.dist(*params)
        new_act = policy.sample()

        q_now = self.critic(obs, new_act)

        with torch.no_grad():
            v_targ = q_now + self.alpha * policy.log_prob(new_act)

        self.v_optim.zero_grad()
        v_pred = self.value_net(obs)
        v_loss = (v_pred - v_targ).pow(2.).mean().mul(0.5)
        v_loss.backward()
        self.v_optim.step()

        self.a_optim.zero_grad()

        # new_act = new_act.detach() <- detach를 하면 잘안됨.

        actor_loss = self.alpha * policy.log_prob(new_act) - q_now
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.a_optim.step()
        self.update_target()

    def update_target(self):
        for w, w_target in zip(self.value_net.parameters(), self.value_net_target.parameters()):
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

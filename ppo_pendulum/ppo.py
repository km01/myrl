from abc import abstractmethod

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, distribution_type):
        super().__init__()
        self._distribution_type = distribution_type

    def forward(self, x):
        params = []
        value = None
        return params, value

    @abstractmethod
    def critic(self, x):
        pass  # return value

    @abstractmethod
    def policy(self, x):
        pass  # return params

    @abstractmethod
    def response(self, x, mode=False):
        pass  # return act, params

    @abstractmethod
    def to_real_action(self, act):
        pass

    @torch.no_grad()
    def sample_n_step(self, ob_env_now, envs, n):
        trans = []
        probs = []
        for _ in range(n):
            ob_env = envs.reset() if ob_env_now is None else ob_env_now
            x = torch.FloatTensor(ob_env)
            act, params = self.response(x)

            ob_env_now, rew, done, _ = envs.step(self.to_real_action(act))

            trans.append([ob_env, act, rew, done])
            probs.append([par.numpy() for par in params])

        trans = [np.stack(val) for val in zip(*trans)]
        probs = [np.stack(val) for val in zip(*probs)]
        return ob_env_now, trans, probs

    @torch.no_grad()
    def compute_gae(self, ob_env, rew, done, ob_env_next, gamma=0.95, tau=0.9):
        ob_env = np.append(ob_env, [ob_env_next], axis=0)
        ob_env = torch.FloatTensor(ob_env)
        rew = torch.FloatTensor(rew)
        done = torch.BoolTensor(done)
        values = self.critic(ob_env)
        values = values.squeeze(-1)
        next_v = values[-1]
        values = values[:-1]
        g_a_e = []
        gae = torch.zeros_like(next_v)
        for r, d, v in reversed(list(zip(rew, done, values))):
            gae = r - v + gamma * (next_v + tau * gae).masked_fill(d, 0.)
            g_a_e.insert(0, gae)
            next_v = v

        g_a_e = torch.stack(g_a_e, dim=0).unsqueeze(-1).numpy()
        values = values.unsqueeze(-1).numpy()
        return g_a_e, values

    def ppo_train(self, memory, batch_size, ppo_epoch, solver, clip_param, w_actor, w_entropy, max_grad_norm):
        action_loss_mean = 0.
        value_loss_mean = 0.
        for _ in range(ppo_epoch):
            batch = memory.sample(batch_size)
            ob_env, act, old_gae, old_val, *old_params = [torch.FloatTensor(x) for x in batch]

            old_dist = self._distribution_type(*old_params)
            new_params, new_val = self(ob_env)
            new_dist = self._distribution_type(*new_params)

            test_version = False
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
            value_loss = (new_val - critic_targ).pow(2).mean()

            solver.zero_grad()

            neg_entropy = - new_dist.entropy().mean()

            loss = w_actor * action_loss + value_loss + w_entropy * neg_entropy
            loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            solver.step()

            action_loss_mean += action_loss.item() / ppo_epoch
            value_loss_mean += value_loss.item() / ppo_epoch
        return action_loss_mean, value_loss_mean

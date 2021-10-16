import torch
import torch.optim as optim
import torch.nn as nn
from rl_utils import compute_ret


class RppoSolver(object):

    def __init__(self,
                 actor_critic,
                 lr,
                 batch_size,
                 num_batch_epoch,
                 vf_coef,
                 ent_coef,
                 clip_range,
                 gamma,
                 max_grad_norm,
                 vf_clip_range=None,
                 lamda=None):

        self.lr = lr
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.num_batch_epoch = num_batch_epoch
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.gamma = gamma

        self.max_grad_norm = max_grad_norm
        self.vf_clip_range = vf_clip_range
        self.lamda = lamda

    def update(self, a2c, experience):
        data, device = experience.fetch(), a2c.device()

        obs_data = torch.FloatTensor(data['obs']).to(device)
        act_data = torch.FloatTensor(data['act']).to(device)
        hid_data = torch.FloatTensor(data['hid']).to(device)

        rew_data = torch.FloatTensor(data['rew']).to(device)
        val_data = torch.FloatTensor(data['val']).to(device)
        log_prob_data = torch.FloatTensor(data['log_prob']).to(device)

        done_data = torch.BoolTensor(data['done']).to(device)
        first_data = torch.BoolTensor(data['first']).to(device)
        truncated_data = torch.BoolTensor(data['truncated']).to(device)

        seq_len, num_seq = obs_data.size()[0:2]

        ret_data = compute_ret(val_data, rew_data, done_data, truncated_data, self.gamma, self.lamda).detach()
        stats = {'value_loss': 0., 'performance': 0., 'entropy': 0.}

        for i in range(self.num_batch_epoch):
            indices = torch.randperm(num_seq)[: self.batch_size]

            batch = {'obs': obs_data[:, indices].clone().detach(),
                     'act': act_data[:, indices].clone().detach(),
                     'val': val_data[:, indices].clone().detach(),
                     'hid': hid_data[:, indices].clone().detach(),
                     'first': first_data[:, indices].clone().detach(),
                     'ret': ret_data[:, indices].clone().detach(),
                     'log_prob': log_prob_data[:, indices].clone().detach()}

            batch_adv = batch['ret'] - batch['val']
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-7)

            self.optimizer.zero_grad()

            policy_param, val, hid = a2c.rollout(seq_obs=batch['obs'],
                                                 h=batch['hid'][0],
                                                 seq_first=batch['first'])

            policy = a2c.policy_class(policy_param)

            ratio = torch.exp(policy.log_prob(batch['act']) - batch['log_prob'])
            surr1 = batch_adv * ratio
            surr2 = batch_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            performance = torch.min(surr1, surr2)
            entropy = - policy.log_prob(policy.sample())

            val_error = (batch['ret'] - val).pow(2).mul(0.5)

            if self.vf_clip_range is not None:
                val_clipped = batch['val'] + (val - batch['val']).clamp(-self.vf_clip_range, self.vf_clip_range)
                val_error_clipped = (batch['ret'] - val_clipped).pow(2).mul(0.5)
                val_error = torch.max(val_error, val_error_clipped)

            performance = performance.mean()
            val_error = val_error.mean()
            entropy = entropy.mean()

            loss = self.vf_coef * val_error - (performance + self.ent_coef * entropy)
            loss.backward()

            nn.utils.clip_grad_norm_(a2c.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats['value_loss'] += val_error.item() / self.num_batch_epoch
            stats['performance'] += performance.item() / self.num_batch_epoch
            stats['entropy'] += entropy.item() / self.num_batch_epoch

        return stats

    def update_calibrate(self, a2c, experience):

        #  매 업데이트 마다 value, log_prob, hidden state 데이터를 최근의 모델에서 얻은 결과로 덮어씌운다.
        #  https://kam.al/blog/ppo_stale_states/

        data, device = experience.fetch(), a2c.device()

        obs_data = torch.FloatTensor(data['obs']).to(device)
        act_data = torch.FloatTensor(data['act']).to(device)
        hid_data = torch.FloatTensor(data['hid']).to(device)

        rew_data = torch.FloatTensor(data['rew']).to(device)
        val_data = torch.FloatTensor(data['val']).to(device)
        log_prob_data = torch.FloatTensor(data['log_prob']).to(device)

        done_data = torch.BoolTensor(data['done']).to(device)
        first_data = torch.BoolTensor(data['first']).to(device)
        truncated_data = torch.BoolTensor(data['truncated']).to(device)

        seq_len, num_seq = obs_data.size()[0:2]

        stats = {'value_loss': 0., 'performance': 0., 'entropy': 0.}

        for i in range(self.num_batch_epoch):
            indices = torch.randperm(num_seq)[: self.batch_size]

            batch = {'obs': obs_data[:, indices].clone().detach(),
                     'act': act_data[:, indices].clone().detach(),
                     'rew': rew_data[:, indices].clone().detach(),
                     'val': val_data[:, indices].clone().detach(),
                     'hid': hid_data[:, indices].clone().detach(),
                     'first': first_data[:, indices].clone().detach(),
                     'log_prob': log_prob_data[:, indices].clone().detach(),
                     'done': done_data[:, indices].clone().detach(),
                     'truncated': truncated_data[:, indices].clone().detach()}

            ret = compute_ret(batch['val'],
                              batch['rew'],
                              batch['done'],
                              batch['truncated'],
                              self.gamma,
                              self.lamda).detach()

            adv = ret - batch['val']
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            self.optimizer.zero_grad()

            policy_param, val, hid = a2c.rollout(seq_obs=batch['obs'],
                                                 h=batch['hid'][0],
                                                 seq_first=batch['first'])
            policy = a2c.policy_class(policy_param)

            log_prob = policy.log_prob(batch['act'])
            ratio = torch.exp(log_prob - batch['log_prob'])
            surr1 = adv * ratio
            surr2 = adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            performance = torch.min(surr1, surr2)
            entropy = - policy.log_prob(policy.sample())

            val_error = (ret - val).pow(2).mul(0.5)

            if self.vf_clip_range is not None:
                val_clipped = batch['val'] + (val - batch['val']).clamp(-self.vf_clip_range, self.vf_clip_range)
                val_error_clipped = (ret - val_clipped).pow(2).mul(0.5)
                val_error = torch.max(val_error, val_error_clipped)

            performance = performance.mean()
            val_error = val_error.mean()
            entropy = entropy.mean()

            loss = self.vf_coef * val_error - (performance + self.ent_coef * entropy)
            loss.backward()

            nn.utils.clip_grad_norm_(a2c.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats['value_loss'] += val_error.item() / self.num_batch_epoch
            stats['performance'] += performance.item() / self.num_batch_epoch
            stats['entropy'] += entropy.item() / self.num_batch_epoch

            val_data[:, indices].copy_(val.clone().detach())
            log_prob_data[:, indices].copy_(log_prob.clone().detach())
            hid_data[1:, indices].copy_(hid[:-1].clone().detach())

        return stats

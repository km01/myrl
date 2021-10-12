import torch
import torch.optim as optim
import torch.nn as nn
from rl_utils import compute_ret


class PPOSolver(object):

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

    def update(self, actor_critic, experience):

        data, device = experience.fetch(), actor_critic.device()

        data['obs'] = torch.FloatTensor(data['obs']).to(device)
        data['act'] = torch.FloatTensor(data['act']).to(device)
        data['val'] = torch.FloatTensor(data['val']).to(device)
        data['log_prob'] = torch.FloatTensor(data['log_prob']).to(device)
        data['rew'] = torch.FloatTensor(data['rew']).to(device)
        data['done'] = torch.BoolTensor(data['done']).to(device)
        data['truncated'] = torch.BoolTensor(data['truncated']).to(device)

        data['ret'] = compute_ret(data['val'],
                                  data['rew'],
                                  data['done'],
                                  data['truncated'],
                                  self.gamma,
                                  self.lamda).detach()

        data['obs'] = data['obs'].flatten(0, 1)
        data['act'] = data['act'].flatten(0, 1)
        data['val'] = data['val'].flatten(0, 1)
        data['ret'] = data['ret'].flatten(0, 1)
        data['log_prob'] = data['log_prob'].flatten(0, 1)

        num_data = data['val'].size()[0]

        stats = {'value_loss': 0., 'performance': 0., 'entropy': 0.}

        for i in range(self.num_batch_epoch):

            indices = torch.randperm(num_data)[: self.batch_size]

            obs_data = data['obs'][indices].clone().detach()
            act_data = data['act'][indices].clone().detach()
            val_data = data['val'][indices].clone().detach()
            ret_data = data['ret'][indices].clone().detach()
            log_prob_data = data['log_prob'][indices].clone().detach()

            adv = (ret_data - val_data)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            self.optimizer.zero_grad()

            policy, val = actor_critic(obs_data)

            ratio = torch.exp(policy.log_prob(act_data) - log_prob_data)
            surr1 = adv * ratio
            surr2 = adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            performance = torch.min(surr1, surr2).mean()
            entropy = - policy.log_prob(policy.sample()).mean()

            val_error = (ret_data - val).pow(2).mul(0.5)

            if self.vf_clip_range is not None:
                val_clipped = val_data + (val - val_data).clamp(-self.vf_clip_range, self.vf_clip_range)
                val_error_clipped = (ret_data - val_clipped).pow(2).mul(0.5)
                val_error = torch.max(val_error, val_error_clipped).mean()

            loss = self.vf_coef * val_error - (performance + self.ent_coef * entropy)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), self.max_grad_norm)

            self.optimizer.step()

            stats['value_loss'] += val_error.item() / self.num_batch_epoch
            stats['performance'] += performance.item() / self.num_batch_epoch
            stats['entropy'] += entropy.item() / self.num_batch_epoch

        return stats

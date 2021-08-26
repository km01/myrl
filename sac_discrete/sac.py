import copy
import torch
from utils import soft_update, Cat
from model import Actor, Critic, QNet, ValueNet
import torch.nn.functional as F
import torch.distributions as D


class SAC(object):
    def __init__(self, input_size, action_size, gamma, tau, alpha, hidden_size, lr, device):

        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.lr, self.device = lr, device

        self.policy = Actor(input_size, hidden_size, action_size).to(self.device)
        self.critic = QNet(input_size, hidden_size, action_size).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

    @torch.no_grad()
    def select_action(self, obs, sample=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        policy = self.policy(obs)
        action = Cat(raw_base=policy).sample(onehot=False, sample=sample)
        action = action.cpu().numpy()[0]
        return action

    def update_parameters(self, batch):
        obs, act, rew, done, obs_next = batch
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.LongTensor(act).unsqueeze(-1).to(self.device)
        rew = torch.FloatTensor(rew).unsqueeze(-1).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(-1).to(self.device)
        obs_next = torch.FloatTensor(obs_next).to(self.device)

        with torch.no_grad():
            next_policy = Cat(raw_base=self.policy(obs_next))
            next_q = torch.min(*self.critic_target(obs_next))
            next_eval = (next_policy.probs * next_q).sum(dim=-1, keepdim=True)
            next_entr = - (next_policy.probs * next_policy.logits).sum(dim=-1, keepdim=True)
            next_v = (next_eval + self.alpha * next_entr).masked_fill(done, 0.)
            q_targ = rew + self.gamma * next_v

        self.critic_optim.zero_grad()
        q1, q2 = self.critic(obs)

        q_pred = torch.min(q1, q2).detach()
        q1, q2 = q1.gather(dim=-1, index=act), q2.gather(dim=-1, index=act)
        critic_loss = (q1 - q_targ).pow(2.).mul(0.5) + (q2 - q_targ).pow(2.).mul(0.5)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            critic_loss = (torch.min(q1, q2) - q_targ).pow(2.).mul(0.5).mean()

        self.policy_optim.zero_grad()
        policy = Cat(raw_base=self.policy(obs))
        policy_entr = - (policy.probs.detach() * policy.logits).sum(dim=-1).mean()
        policy_eval = (policy.probs * q_pred).sum(dim=-1).mean()
        policy_loss = self.alpha * policy_entr - policy_eval
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

        loss_info = {'critic_loss': critic_loss.item(),
                     'policy_loss': policy_loss.item(),
                     'policy_entr': policy_entr.item()}

        return loss_info











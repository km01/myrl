import copy
import torch
from utils import soft_update, TanhGaussian
from model import Actor, Critic, ValueNetwork


class SAC(object):
    def __init__(self, input_size, action_size, gamma, tau, alpha, hidden_size, lr, device):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device

        self.policy = Actor(input_size, hidden_size, action_size).to(self.device)
        self.critic = Critic(input_size, hidden_size, action_size).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

    def select_action(self, obs, sample=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        policy = TanhGaussian(*self.policy(obs))
        action = policy.sample(sample)
        action = action.detach().cpu().numpy()[0]
        return action

    def update_parameters(self, batch):

        obs, act, rew, done, obs_next = batch
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).unsqueeze(-1).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(-1).to(self.device)
        obs_next = torch.FloatTensor(obs_next).to(self.device)

        with torch.no_grad():
            next_policy = TanhGaussian(*self.policy(obs_next))
            next_action = next_policy.sample()
            next_log_pi = next_policy.log_prob(next_action).sum(dim=-1, keepdim=True)
            next_q = torch.min(*self.critic_target(obs_next, next_action))
            next_v = (next_q - self.alpha * next_log_pi).masked_fill(done, 0.)
            q_targ = rew + self.gamma * next_v

        self.critic_optim.zero_grad()
        q1, q2 = self.critic(obs, act)
        critic_loss = (q1 - q_targ).pow(2.).mul(0.5) + (q2 - q_targ).pow(2.).mul(0.5)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            critic_loss = (torch.min(q1, q2) - q_targ).pow(2).mul(0.5).mean()

        self.policy_optim.zero_grad()
        policy = TanhGaussian(*self.policy(obs))
        action = policy.sample()
        log_pi = policy.log_prob(action, param_grad=False).sum(dim=-1, keepdim=True)
        q_pred = torch.min(*self.critic(obs, action))
        policy_loss = self.alpha * log_pi - q_pred
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

        loss_info = {'critic_loss': critic_loss.item(),
                     'policy_loss': policy_loss.item(),
                     'policy_entropy': -log_pi.mean().item()}

        return loss_info


class SACV(object):
    def __init__(self, input_size, action_size, gamma, tau, alpha, hidden_size, lr, device):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device

        self.policy = Actor(input_size, hidden_size, action_size).to(self.device)
        self.critic = Critic(input_size, hidden_size, action_size).to(self.device)
        self.value = ValueNetwork(input_size, hidden_size).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.value_target = copy.deepcopy(self.value)
        self.value_target.requires_grad_(False)

    def select_action(self, obs, sample=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        policy = TanhGaussian(*self.policy(obs))
        action = policy.sample(sample)
        action = action.detach().cpu().numpy()[0]
        return action

    def update_parameters(self, batch):

        obs, act, rew, done, obs_next = batch
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).unsqueeze(-1).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(-1).to(self.device)
        obs_next = torch.FloatTensor(obs_next).to(self.device)

        with torch.no_grad():
            next_v = self.value_target(obs_next).masked_fill(done, 0.)
            q_targ = rew + self.gamma * next_v

        self.critic_optim.zero_grad()
        q1, q2 = self.critic(obs, act)
        critic_loss = (q1 - q_targ).pow(2.).mul(0.5) + (q2 - q_targ).pow(2.).mul(0.5)
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            critic_loss = (torch.min(q1, q2) - q_targ).pow(2).mul(0.5).mean()

        self.policy_optim.zero_grad()
        policy = TanhGaussian(*self.policy(obs))
        action = policy.sample()
        log_pi = policy.log_prob(action, param_grad=False).sum(dim=-1, keepdim=True)

        action_value = torch.min(*self.critic(obs, action))

        with torch.no_grad():
            v_targ = action_value - self.alpha * log_pi

        self.value_optim.zero_grad()
        v = self.value(obs)
        value_loss = (v - v_targ).pow(2.).mul(0.5)
        value_loss = value_loss.mean()
        value_loss.backward()
        self.value_optim.step()

        policy_loss = self.alpha * log_pi - action_value
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.value_target, self.value, self.tau)
        loss_info = {'critic_loss': critic_loss.item(),
                     'policy_loss': policy_loss.item(),
                     'value_loss': value_loss.item(),
                     'policy_entropy': -log_pi.mean().item()}
        return loss_info

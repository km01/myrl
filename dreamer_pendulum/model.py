import torch
import torch.nn as nn
from utils import NormalPredictor, BernoulliPredictor


class Transition(nn.Module):
    def __init__(self, h_size, z_size, a_size):
        super().__init__()
        self.h_size = h_size
        self.aggr = nn.Sequential(
            nn.Linear(z_size + a_size, h_size),
            nn.ELU()
        )
        self.core = nn.GRUCell(h_size, h_size)

    def initial_belief(self, batch_size):
        return torch.zeros((batch_size, self.h_size), dtype=torch.float)

    def forward(self, state, action, prev_belief):
        input = torch.cat([state, action], dim=-1)
        input = self.aggr(input)
        belief = self.core(input, prev_belief)
        return belief


class RSSM(nn.Module):
    def __init__(self, x_size, h_size, z_size, a_size):
        super().__init__()
        self.transition = Transition(h_size, z_size, a_size)
        self.z_q_dist = NormalPredictor(x_size + h_size, h_size, z_size)
        self.z_p_dist = NormalPredictor(h_size, h_size, z_size)

    def forward(self, obs_seq, act_seq):

        beliefs = [self.transition.initial_belief(batch_size=obs_seq.size(1))]
        z_q_params, z_p_params, z_qs = [], [], []

        for obs, act in zip(obs_seq, act_seq):

            input = torch.cat([obs, beliefs[-1]], dim=-1)
            z_q_param = self.z_q_dist(input)
            z_q = NormalPredictor.sample(*z_q_param)
            z_p_param = self.z_p_dist(beliefs[-1])
            belief = self.transition(z_q, act, beliefs[-1])

            z_q_params.append(z_q_param)
            z_p_params.append(z_p_param)
            z_qs.append(z_q)
            beliefs.append(belief)

        beliefs = torch.stack(beliefs, dim=0)
        z_q_params = [torch.stack(item, dim=0) for item in zip(*z_q_params)]
        z_p_params = [torch.stack(item, dim=0) for item in zip(*z_p_params)]
        z_qs = torch.stack(z_qs, dim=0)
        return beliefs, z_q_params, z_p_params, z_qs


class Model(nn.Module):
    def __init__(self, x_size, h_size, z_size, a_size):
        super().__init__()
        self.rssm = RSSM(x_size, h_size, z_size, a_size)
        self.obs_p = NormalPredictor(z_size + h_size, h_size, x_size)
        self.rew_p = NormalPredictor(z_size + h_size, h_size, 1)
        self.gamma_p = BernoulliPredictor(z_size + h_size, h_size, 1)

    def forward(self, obs_seq, act_seq):
        num_seq, batch_size = obs_seq.size(0), obs_seq.size(1)
        belief, z_q_param, z_p_param, z_q = self.rssm(obs_seq, act_seq)
        z_q_flatten = z_q.view(num_seq * batch_size, -1)
        belief_prev = belief[: -1].view(num_seq * batch_size, -1)
        belief_next = belief[1:].view(num_seq * batch_size, -1)

        obs_param = self.obs_p(torch.cat([z_q_flatten, belief_prev], dim=-1))
        rew_param = self.rew_p(torch.cat([z_q_flatten, belief_next], dim=-1))
        gamma_param = self.gamma_p(torch.cat([z_q_flatten, belief_next], dim=-1))

        obs_param = [item.view(num_seq, batch_size, -1) for item in obs_param]
        rew_param = [item.view(num_seq, batch_size, -1) for item in rew_param]
        gamma_param = [item.view(num_seq, batch_size, -1) for item in gamma_param]

        return z_q, z_q_param, z_p_param, obs_param, rew_param, gamma_param

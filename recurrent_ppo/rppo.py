import torch
import torch.nn as nn
from dynamics import GruDynamics
from model import ActorCritic, MLP
from distributions import Gaussian


class Rppo(nn.Module):
    def __init__(self,

                 obs_size,
                 obs_enc_size,

                 policy_param_size,
                 policy_class,

                 belief_size,
                 a2c_input_size,
                 a2c_hidden_size,
                 a2c_shared_layers,
                 a2c_actor_layers,
                 a2c_critic_layers):

        super().__init__()

        self.obs_enc = MLP([obs_size, obs_enc_size, obs_enc_size, obs_enc_size])

        self.agg = MLP([obs_enc_size + belief_size, a2c_input_size], nn.ELU())

        self.dynamics = GruDynamics(obs_enc_size, belief_size)

        self.a2c = ActorCritic(input_size=a2c_input_size,
                               action_param_size=policy_param_size,
                               policy_class=policy_class,
                               hidden_size=a2c_hidden_size,
                               num_actor_layers=a2c_actor_layers,
                               num_critic_layers=a2c_critic_layers,
                               num_shared_layers=a2c_shared_layers)

        self.policy_class = self.a2c.policy_class

    def device(self):
        return next(self.parameters()).device

    def initial_state(self, batch_size, requires_grad=True):
        h = self.dynamics.initial_state(batch_size).to(self.device())
        h.requires_grad_(requires_grad)
        return h

    def masked_fill_initial_state(self, h, initial_mask):
        return self.dynamics.masked_fill_initial_state(h, initial_mask)

    @torch.no_grad()
    def step(self, obs, hid, det=False):
        obs = torch.FloatTensor(obs).to(self.device())
        obs = self.obs_enc(obs)

        agg = self.agg(torch.cat([obs, hid], dim=-1))
        policy_param, val_pred = self.a2c(agg)
        policy = self.a2c.policy_class(policy_param)
        act = policy.sample(det)
        log_prob = policy.log_prob(act)

        _, h_next = self.dynamics(obs, hid)
        info = {'hid': hid.detach().cpu().numpy(),
                'val': val_pred.detach().cpu().numpy(),
                'act': act.detach().cpu().numpy(),
                'log_prob': log_prob.detach().cpu().numpy()}

        return policy.as_env(act), info, h_next

    def rollout(self, seq_obs, h, seq_first):
        seq_len, batch_size = seq_obs.size()[0:2]
        seq_obs = self.obs_enc(seq_obs)

        hid = [h]
        for t in range(seq_len):
            h = self.masked_fill_initial_state(h, seq_first[t])
            _, h = self.dynamics(seq_obs[t], h)
            hid.append(h)

        hid = torch.stack(hid, dim=0)  # h[0:T]
        agg = self.agg(torch.cat([seq_obs, hid[:-1]], dim=-1))
        policy_param, val_pred = self.a2c(agg)
        hid = hid[1:]   # h[1:T]
        return policy_param, val_pred, hid

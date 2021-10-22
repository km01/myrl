import torch
import torch.nn as nn
from dynamics import GruDynamics
from model import ActorCritic, MLP


class Rppo(nn.Module):
    def __init__(self,

                 obs_size,
                 obs_enc_size,

                 act_size,
                 act_enc_size,
                 policy_param_size,
                 policy_class,

                 belief_size,
                 state_size,

                 a2c_hidden_size,
                 a2c_shared_layers,
                 a2c_actor_layers,
                 a2c_critic_layers):

        super().__init__()

        self.obs_enc = MLP([obs_size, obs_enc_size, obs_enc_size, obs_enc_size])
        self.act_enc = MLP([act_size, act_enc_size, act_enc_size, act_enc_size])

        self.agent_state = MLP([obs_enc_size + belief_size, obs_enc_size + belief_size, state_size])

        self.sa_enc = MLP([state_size + act_enc_size, belief_size, belief_size])

        self.dynamics = GruDynamics(belief_size, belief_size)

        self.a2c = ActorCritic(input_size=state_size,
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

        state = self.agent_state(torch.cat([obs, hid], dim=-1))
        policy_param, val_pred = self.a2c(state)
        policy = self.a2c.policy_class(policy_param)
        act = policy.sample(det)
        log_prob = policy.log_prob(act)

        if self.policy_class.name == 'Gaussian':
            act_enc = self.act_enc(act.tanh())
        else:
            act_enc = self.act_enc(act)

        sa_enc = self.sa_enc(torch.cat([state, act_enc], dim=-1))
        _, hid_next = self.dynamics(sa_enc, hid)

        info = {'hid': hid.detach().cpu().numpy(),
                'val': val_pred.detach().cpu().numpy(),
                'act': act.detach().cpu().numpy(),
                'log_prob': log_prob.detach().cpu().numpy()}

        return policy.as_env(act), info, hid_next

    def rollout(self, seq_obs, seq_act, seq_first, hid0):
        seq_len, batch_size = seq_obs.size()[0:2]
        seq_obs = self.obs_enc(seq_obs)

        if self.policy_class.name == 'Gaussian':
            seq_act = self.act_enc(seq_act.tanh())
        else:
            seq_act = self.act_enc(seq_act)

        hid, state = [hid0], []
        h = hid[0]
        for t in range(seq_len):
            h = self.masked_fill_initial_state(h, seq_first[t])
            s = self.agent_state(torch.cat([seq_obs[t], h], dim=-1))
            sa = self.sa_enc(torch.cat([s, seq_act[t]], dim=-1))
            _, h = self.dynamics(sa, h)
            state.append(s)
            hid.append(h)

        state = torch.stack(state, dim=0)  # s[0:T-1]
        hid = torch.stack(hid, dim=0)  # h[0:T]
        policy_param, val_pred = self.a2c(state)
        hid = hid[1:]   # h[1:T]
        return policy_param, val_pred, state, hid

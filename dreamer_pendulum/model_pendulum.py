import torch
from model import Model
import torch.optim as optim
from utils import NormalPredictor, BernoulliPredictor


class ModelPendulum(object):

    def __init__(self, h_size, z_size, lr):
        self.x_size = 3
        self.z_size = z_size
        self.h_size = h_size
        self.a_size = 1
        self.model = Model(self.x_size, h_size, z_size, self.a_size)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, batch):
        self.optim.zero_grad()

        obs, act, rew, done, invalid = batch
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew).unsqueeze(-1)
        rew = (rew + 8.) / 8.

        done = torch.BoolTensor(done).unsqueeze(-1)
        discount = torch.logical_not(done).float()

        invalid = torch.BoolTensor(invalid)

        z_q, z_q_param, z_p_param, obs_param, rew_param, gamma_param = self.model(obs, act)

        log_p_z = NormalPredictor.log_prob(z_q, *z_p_param).sum(dim=-1) / self.z_size
        log_q_z = NormalPredictor.log_prob(z_q, *z_q_param).sum(dim=-1) / self.z_size
        log_p_x = NormalPredictor.log_prob(obs, *obs_param).sum(dim=-1) / self.x_size
        log_p_r = NormalPredictor.log_prob(rew, *rew_param).sum(dim=-1)
        log_p_dis = BernoulliPredictor.log_prob(discount, *gamma_param).sum(dim=-1)

        log_p_z = log_p_z.masked_fill(invalid, 0.).mean()
        log_q_z = log_q_z.masked_fill(invalid, 0.).mean()
        log_p_x = log_p_x.masked_fill(invalid, 0.).mean()
        log_p_r = log_p_r.masked_fill(invalid, 0.).mean()
        log_p_dis = log_p_dis.masked_fill(invalid, 0.).mean()

        loss = (log_p_x + log_p_r + log_p_dis + 0.1 * log_p_z - 0.05 * log_q_z).mul(-1.)
        loss.backward()
        self.optim.step()

        p_z_loss = - log_p_z.item()
        q_z_loss = log_q_z.item()
        p_x_loss = log_p_x.item()
        p_r_loss = log_p_r.item()
        p_dis_loss = log_p_dis.item()
        print(p_z_loss, q_z_loss, p_x_loss, p_r_loss, p_dis_loss)
        return p_z_loss, q_z_loss, p_x_loss, p_r_loss, p_dis_loss

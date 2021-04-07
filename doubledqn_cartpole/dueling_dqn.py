import torch
import torch.nn as nn
import numpy as np
import copy


class Model(nn.Module):

    def __init__(self, net, n_a):
        super().__init__()
        self.net = net  # policy network
        self.n_a = n_a
        self._net = copy.deepcopy(net)  # fixed action-value (for greedy policy by old q value)
        self.update()

    def update(self):
        for theta_old, theta_new in zip(self._net.parameters(), self.net.parameters()):
            theta_old.data.copy_(theta_new.data)

    def forward(self, x):
        return self.net(x)

    def parameters(self, recurse: bool = True):
        return self.net.parameters()

    def response(self, x, eps=None):
        with torch.no_grad():
            a = self.net(x).argmax(dim=1, keepdim=False)
            if eps is not None:
                for i in range(len(x)):
                    if eps > np.random.rand():
                        a[i] = np.random.randint(self.n_a)
            a = a.numpy()
        return a

    def train_model(self, batch, optimizer, gamma, loss_fn):
        s, a, r, next_s, next_t = batch.s, batch.a, batch.r, batch.ns, batch.nt

        q_s, q_ns = self(torch.cat([s, next_s], dim=0)).chunk(2, dim=0)
        argmax_q_ns = q_ns.argmax(dim=1, keepdim=True)
        q_s = q_s.gather(1, a)

        with torch.no_grad():
            backed_up = self._net(next_s)

        backed_up = backed_up.gather(1, argmax_q_ns).masked_fill(next_t, 0.)
        trg = r + gamma * backed_up
        loss = loss_fn(q_s, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

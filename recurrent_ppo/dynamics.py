import torch
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()

    def initial_state(self, batch_size):
        raise NotImplementedError

    def masked_fill_initial_state(self, h, done):
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError


class GruDynamics(Dynamics):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.dynamics = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def masked_fill_initial_state(self, h, done):
        if not torch.is_tensor(done):
            done = torch.BoolTensor(done).to(h.device)

        done = done.unsqueeze(-1).expand(-1, self.hidden_size)
        return h.masked_fill(done, 0.)

    def forward(self, x, h):
        assert x.ndim == h.ndim and x.size()[0] == h.size()[0], 'invalid x and h'
        h_next = self.dynamics(x, h)
        return h_next, h_next

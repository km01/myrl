import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('s', 'a', 'r', 'ns', 'nt'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, s, a, r, ns, nt):
        self.memory.append(Transition(s, a, r, ns, nt))

    def push_many(self, s_list, a_list, r_list, ns_list, nt_list):
        for s, a, r, ns, nt in zip(s_list, a_list, r_list, ns_list, nt_list):
            self.push(s, a, r, ns, nt)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

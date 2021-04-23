from collections import deque
import random
import numpy as np


class Memory(object):
    def __init__(self, size):
        self.size = size
        self.queue = deque(maxlen=self.size)

    def __len__(self):
        return len(self.queue)

    def push(self, *args):
        self.queue.append([*args])

    def sample(self, batch_size):
        items = random.sample(self.queue, batch_size)
        items = [np.stack(x) for x in zip(*items)]
        return items

    def push_many(self, *args_list):
        for args in zip(*args_list):
            self.push(*args)

    def push_n_step_samples(self, *arg_list):
        arg_list = [arg.reshape((-1, *arg.shape[2:])) for arg in arg_list]
        self.push_many(*arg_list)





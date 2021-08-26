from collections import deque
import random
import numpy as np


class Memory(object):
    def __init__(self, maxlen):
        self.queue = deque(maxlen=maxlen)

    @property
    def maxlen(self):
        return self.queue.maxlen

    def __len__(self):
        return len(self.queue)

    def sample(self, batch_size):
        items = random.sample(self.queue, batch_size)
        return [np.stack(x) for x in zip(*items)]

    def push(self, items):
        for item in zip(*items):
            self.queue.append(item)

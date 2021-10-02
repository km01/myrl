import numpy as np


class OnlineRolloutStorage(object):
    def __init__(self, maxlen):
        self.cache = {}
        self.maxlen = maxlen
        self.ptr = 0
        self.max_pull = 0

    def __len__(self):
        return self.max_pull

    def _allocate(self, dic):
        for key, item in dic.items():
            if key not in self.cache:
                item_memory = np.zeros_like(item)
                item_memory = np.expand_dims(item_memory, axis=0)
                item_memory = np.repeat(item_memory, self.maxlen, axis=0)
                self.cache[key] = item_memory

    def push(self, dic):
        if len(self.cache) == 0:
            self._allocate(dic)

        for key, item in dic.items():
            self.cache[key][self.ptr] = item

        self.ptr = (self.ptr + 1) % self.maxlen
        if self.max_pull < self.maxlen:
            self.max_pull += 1

    def fetch(self, length=None, reverse=False):
        if length is None:
            length = self.max_pull

        assert length <= self.max_pull
        indices = list(range(self.ptr, self.maxlen)) + list(range(0, self.ptr))
        indices = indices[-length:]
        if reverse is True:
            indices.reverse()

        items = {}
        for key, item in self.cache.items():
            items[key] = self.cache[key][indices]

        return items


if __name__ == '__main__':
    n_step = 17
    experience = OnlineRolloutStorage(maxlen=n_step)
    for t in range(41):
        experience.push({'observation': np.array([t]),
                         'done': np.array(t % 2).astype(np.bool)})

    data = experience.fetch(16)
    print(data['observation'])

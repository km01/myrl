from collections import deque
import numpy as np


class Rollout(object):
    def __init__(self):
        self.obs = []
        self.act = []
        self.rew = []
        self.last_done = False

    def __len__(self):
        return len(self.act)

    def push(self, obs, act, rew):
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)

    def register(self, last_done):
        self.last_done = last_done
        self.obs = np.array(self.obs)
        self.act = np.array(self.act)
        self.rew = np.array(self.rew)

    def fetch(self, num_seq, start_index=None):

        rollout_len = self.__len__()

        if start_index is None:
            start_index = np.random.randint(0, rollout_len)

        else:
            assert rollout_len > start_index

        obs = np.zeros((num_seq, *self.obs.shape[1:]), dtype=self.obs.dtype)
        act = np.zeros((num_seq, *self.act.shape[1:]), dtype=self.act.dtype)

        rew = np.zeros((num_seq, *self.rew.shape[1:]), dtype=self.rew.dtype)
        done = np.zeros((num_seq, ), dtype=np.bool)
        invalid = np.ones((num_seq,), dtype=np.bool)

        head = start_index
        tail = head + num_seq

        if tail > rollout_len:
            tail = rollout_len

        valid_len = tail - head

        obs[:valid_len] = self.obs[head:tail]
        act[:valid_len] = self.act[head:tail]
        rew[:valid_len] = self.rew[head:tail]
        done[:valid_len] = False
        invalid[:valid_len] = False

        if tail == rollout_len and self.last_done:
            done[valid_len-1] = True

        return obs, act, rew, done, invalid


class RolloutMemory(object):
    def __init__(self, num_stream, rollout_maxlen, max_queue_len):
        self.num_stream = num_stream
        self.rollout_maxlen = rollout_maxlen
        self.max_queue_len = max_queue_len
        self.queue = deque(maxlen=max_queue_len)
        self.buffers = [Rollout() for _ in range(num_stream)]

    def push(self, obs, act, rew, done):
        for i, o, a, r, d in zip(range(self.num_stream), obs, act, rew, done):
            self.buffers[i].push(o, a, r)
            if d.item():  # if terminal
                self.buffers[i].register(last_done=True)
                self.queue.append(self.buffers[i])
                self.buffers[i] = Rollout()

            elif len(self.buffers[i]) == self.rollout_maxlen:
                self.buffers[i].register(last_done=False)
                self.queue.append(self.buffers[i])
                self.buffers[i] = Rollout()
                self.buffers[i].push(o, a, r)

    def __len__(self):
        return len(self.queue)

    def fetch_data(self, batch_size, num_seq):
        batch = []
        for _ in range(batch_size):
            data_index = np.random.randint(0, self.__len__())
            obs, act, rew, done, invalid = self.queue[data_index].fetch(num_seq)
            batch.append([obs, act, rew, done, invalid])

        batch = [np.stack(item, axis=1) for item in zip(*batch)]
        obs, act, rew, done, invalid = batch
        return obs, act, rew, done, invalid

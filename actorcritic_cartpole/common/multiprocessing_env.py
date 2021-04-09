# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe
import pickle
import cloudpickle
import gym


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    # remote로 할일과 데이터를 받아옴.
    # env에 일을 수행.
    # remote로 결과를 보냄
    parent_remote.close()

    env = env_fn_wrapper.x()

    while True:

        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))

        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)

        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

        else:
            raise NotImplementedError


class VecEnv(object):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def close(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))

        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        ob_s, rew_s, done_s, info_s = zip(*results)
        return np.stack(ob_s), np.stack(rew_s), np.stack(done_s), info_s

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.n_envs

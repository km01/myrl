import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import gym


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        env = gym.wrappers.TimeLimit(env)
        if type(env.action_space) == gym.spaces.box.Box:
            env = gym.wrappers.RescaleAction(env, -1., 1.)

        return env
    return _thunk


def make_vec_env(env_name, num_env, multi_processing=True):
    env = [make_env(env_name) for _ in range(num_env)]
    if multi_processing:
        return SubprocVecEnv(env)
    else:
        return DummyVecEnv(env)


def is_truncated(infos):
    truncated = [info["TimeLimit.truncated"] if "TimeLimit.truncated" in info.keys() else False for info in infos]
    truncated = np.array(truncated)
    return truncated

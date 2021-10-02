import numpy as np
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        env = gym.wrappers.TimeLimit(env)
        if type(env.action_space) == gym.spaces.box.Box:
            env = gym.wrappers.RescaleAction(env, -1., 1.)

        return env
    return _thunk


def is_truncated(infos):
    truncated = [info["TimeLimit.truncated"] if "TimeLimit.truncated" in info.keys() else False for info in infos]
    truncated = np.array(truncated)
    return truncated


def make_vec_env(env_name, num_env):
    env = [make_env(env_name) for _ in range(num_env)]
    return SubprocVecEnv(env)


def make_test_env(env_name, rescaling_action=True):
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env)
    if type(env.action_space) == gym.spaces.box.Box:
        env = gym.wrappers.RescaleAction(env, -1., 1.)

    return env


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, activate=True):
        self.mean = np.zeros((), 'float64')
        self.var = np.ones((), 'float64')
        self.count = 1e-4
        self.activate = activate

    def update(self, x):
        if self.activate:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):

        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def normalize(self, x):
        if self.activate:
            return (x - self.mean) / np.sqrt(self.var)
            # return x / np.sqrt(self.var)
        else:
            return x


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def evaluate(env_name, actor_critic):
    test_gain_avg = 0.
    test_life_avg = 0.
    for _ in range(10):
        test_eval = make_test_env(env_name, rescaling_action=True)
        obs_eval = test_eval.reset()
        gain_eval = 0.
        life_eval = 0
        while True:
            action_eval, agent_info = actor_critic.select_action([obs_eval], sample=False)
            obs_eval, rew_eval, done_eval, _ = test_eval.step(action_eval[0])
            gain_eval += rew_eval
            life_eval += 1
            if done_eval:
                break
        test_eval.close()
        test_gain_avg += gain_eval / 10.
        test_life_avg += life_eval / 10.

    return test_gain_avg, test_life_avg

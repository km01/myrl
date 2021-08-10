import gym
import time
import torch


def pendulum_step(obs, envs, model):
    obs = envs.reset() if obs is None else obs
    act = model.act(obs)
    obs_next, rew, done, _ = envs.step(act * 2.)
    rew = (rew + 8.) / 8.
    infos = [obs, act, rew, done, obs_next]
    return obs_next, infos


def test_env(env_name, model):
    env = gym.make(env_name)
    reward_sum = 0.0
    obs, done = env.reset(), False
    while not done:
        x = torch.FloatTensor(obs).unsqueeze(0)
        a = model.act(x, True)[0]
        obs, rew, done, _ = env.step(2. * a)
        reward_sum += rew

    env.close()
    return reward_sum


def render_simulation(env_name, model):
    env = gym.make(env_name)
    reward_sum = 0.0
    obs, done = env.reset(), False
    while not done:
        x = torch.FloatTensor(obs).unsqueeze(0)
        env.render()
        time.sleep(0.005)
        a = model.act(x, True)[0]
        obs, rew, done, _ = env.step(2. * a)
        reward_sum += rew

    env.close()
    return reward_sum

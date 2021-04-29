import torch
import gym
import time


@torch.no_grad()
def render_simulation(env_name, ddpg):
    env = gym.make(env_name)
    with torch.no_grad():
        obs = env.reset()
        while True:
            env.render()
            time.sleep(0.02)
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = ddpg(x)
            real_action = ddpg.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            if done:
                break
    env.close()


@torch.no_grad()
def test_env(env_name, ddpg):
    env = gym.make(env_name)
    reward_sum = 0.0
    with torch.no_grad():
        obs = env.reset()
        while True:
            x = torch.FloatTensor(obs).unsqueeze(0)
            act = ddpg(x)
            real_action = ddpg.transform_action(act).squeeze(0)
            obs, rew, done, _ = env.step(real_action.numpy())
            reward_sum += rew
            if done:
                break
    env.close()
    return reward_sum

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import time


def plot_array(arr, x_label, y_label):
    arr = np.array(arr)
    avg_100 = []
    for i in range(len(arr)):
        if i < 100:
            avg_100.append(arr[:i + 1].mean())
        else:
            avg_100.append(arr[i - 100:i + 1].mean())
    plt.figure(figsize=(12, 4))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(range(len(arr)), arr)
    plt.plot(range(len(avg_100)), avg_100)
    plt.show()


@torch.no_grad()
def render_simulation(env_name, agent):
    env = gym.make(env_name)
    with torch.no_grad():
        ob = env.reset()
        while True:
            env.render()
            time.sleep(0.02)
            ob_in = torch.FloatTensor([ob])
            act, _ = agent.response(ob_in, True)
            real_action = agent.to_real_action(act).squeeze(0).numpy()
            ob, rew, done, _ = env.step(real_action)
            if done:
                break
    env.close()


@torch.no_grad()
def test_env(env_name, agent):
    env = gym.make(env_name)
    reward_sum = 0.0
    with torch.no_grad():
        ob = env.reset()
        while True:
            ob_in = torch.FloatTensor([ob])
            act, _ = agent.response(ob_in, True)

            real_action = agent.to_real_action(act).squeeze(0).numpy()
            ob, rew, done, _ = env.step(real_action)
            reward_sum += rew
            if done:
                break
    env.close()
    return reward_sum

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


def render_simulation(env_name, actor):
    env = gym.make(env_name)
    with torch.no_grad():
        ob = env.reset()
        while True:
            env.render()
            time.sleep(0.02)
            ob_in = torch.FloatTensor([ob])
            act, _ = actor(ob_in)
            act.clamp_(-2.0, 2.0).squeeze(-1).item()
            ob, rew, done, _ = env.step(act)
            if done:
                break
    env.close()


def test_env(env_name, actor):
    env = gym.make(env_name)
    gain = 0.0
    with torch.no_grad():
        ob = env.reset()
        while True:
            ob_in = torch.FloatTensor([ob])
            act, _ = actor(ob_in)
            act.clamp_(-2.0, 2.0).squeeze(-1).item()
            ob, rew, done, _ = env.step(act)
            gain += rew
            if done:
                break
    env.close()
    return gain

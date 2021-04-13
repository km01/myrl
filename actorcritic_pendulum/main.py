import gym
from common.multiprocessing_env import SubprocVecEnv, make_env
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import Actor, Critic, compute_td
from torch.distributions import Normal
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def plot_array(arr):
    arr = np.array(arr)
    avg_100 = []
    for i in range(len(arr)):
        if i < 100:
            avg_100.append(arr[:i + 1].mean())

        else:
            avg_100.append(arr[i - 100:i + 1].mean())
    plt.figure(figsize=(12, 4))
    plt.xlabel('episode')
    plt.ylabel('duration')

    plt.plot(range(len(arr)), arr)
    plt.plot(range(len(avg_100)), avg_100)
    plt.show()


def render_simulation(env_name_t, actor_t, device_t):
    env_t = gym.make(env_name_t)
    s_ = env_t.reset()
    while True:
        with torch.no_grad():
            a_, _ = actor_t(torch.FloatTensor([s_]).to(device_t))
            a_.clamp_(-2.0 + 1e-7, 2.0 - 1e-7).squeeze(-1).item()
        s_, r_, done_, _ = env_t.step(a_)
        env_t.render()
        time.sleep(0.02)
        if done_:
            break
    env_t.close()


env_name = 'Pendulum-v0'
gamma = 0.9
num_envs = 12  # num_envs 가 크면 오류발생 가능
max_frame = 50000
actor_lr = 0.0003
critic_lr = 0.001
max_grad_norm = 0.7
n_steps = 50
max_episode_steps = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    envs = SubprocVecEnv([make_env(env_name) for i in range(num_envs)])
    envs.set_max_episode_steps(max_episode_steps)
    actor = Actor().to(device)
    critic = Critic().to(device)
    a_solver = optim.Adam(actor.parameters(), lr=actor_lr)
    c_solver = optim.Adam(critic.parameters(), lr=critic_lr)

    frame_count = 0
    rewards = [[0.] for _ in range(num_envs)]
    global_rewards = []

    obs_gotten = None

    while frame_count < max_frame:

        cache = {'obs': [], 'acts': [], 'rews': [], 'dones': []}
        probs_cache = {'mu': [], 'sig': []}

        for _ in range(n_steps):
            obs = envs.reset() if obs_gotten is None else obs_gotten
            obs_in = torch.FloatTensor(obs).to(device)
            mu, sig = actor(obs_in)
            with torch.no_grad():
                a = Normal(mu, sig).sample()
                a.clamp_(-2.0 + 1e-7, 2.0 - 1e-7)

            obs_gotten, rews, dones, _ = envs.step(a)

            for i in range(num_envs):
                rewards[i][-1] += rews[i]
                if dones[i]:
                    global_rewards.append(rewards[i][-1])
                    rewards[i].append(0.)

            cache['obs'].append(obs)
            cache['acts'].append(a)
            cache['rews'].append(rews * 0.1)
            cache['dones'].append(dones)

            probs_cache['mu'].append(mu)
            probs_cache['sig'].append(sig)

        next_obs_in = torch.FloatTensor(obs_gotten).to(device)
        cache['obs'] = torch.FloatTensor(cache['obs']).to(device)
        cache['acts'] = torch.stack(cache['acts'], dim=0)
        probs_cache['mu'] = torch.stack(probs_cache['mu'], dim=0)
        probs_cache['sig'] = torch.stack(probs_cache['sig'], dim=0)
        next_v = critic(next_obs_in).squeeze(-1).detach()
        td_trgs = compute_td(next_v, cache['rews'], cache['dones'], gamma).unsqueeze(-1)
        values = critic(cache['obs'])
        td_res = td_trgs - values
        dist = Normal(probs_cache['mu'], probs_cache['sig'])
        log_prob = dist.log_prob(cache['acts'])
        critic_loss = td_res.pow(2)
        actor_loss = -log_prob * td_res.detach()
        neg_entropy = -dist.entropy()
        loss = critic_loss + actor_loss + 0.00001 * neg_entropy
        loss = loss.sum(dim=[0, 2]).mean()

        a_solver.zero_grad()
        c_solver.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        a_solver.step()
        c_solver.step()

        frame_count += n_steps
        if (frame_count / n_steps) % 10 == 0 and len(global_rewards) > 100:
            avg = float(np.array(global_rewards)[-100:].mean())
            print('[{}/{}]\taverage reward: {}'.format(frame_count, max_frame, avg))

    plot_array(global_rewards)
    render_simulation(env_name, actor, device)


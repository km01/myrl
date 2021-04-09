import gym
import numpy
from common.multiprocessing_env import SubprocVecEnv, make_env
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import Actor, Critic, compute_td
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def plot_duration(durations):
    durations = np.array(durations)
    avg_100 = []
    for i in range(len(durations)):
        if i < 100:
            avg_100.append(durations[:i + 1].mean())

        else:
            avg_100.append(durations[i - 100:i + 1].mean())
    plt.figure(figsize=(12, 4))
    plt.xlabel('episode')
    plt.ylabel('duration')

    plt.plot(range(len(durations)), durations)
    plt.plot(range(len(avg_100)), avg_100)
    plt.show()


def render_simulation(env_name_t, actor_t, device_t):
    env_t = gym.make(env_name_t)
    s_ = env_t.reset()
    while True:
        with torch.no_grad():
            p_ = actor_t(torch.FloatTensor([s_]).to(device_t))
            a_ = p_.argmax(dim=1).squeeze(0).item()
        s_, r_, done_, _ = env_t.step(a_)
        env_t.render()
        time.sleep(0.02)
        if done_:
            break
    env_t.close()


env_name = 'CartPole-v1'
gamma = 0.99
num_envs = 8
PENALTY = -1.0
n_step = 4
max_frame = 50000
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    envs = SubprocVecEnv([make_env(env_name) for i in range(num_envs)])
    net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
    actor = Actor(4, 128, 2).to(device)
    critic = Critic(4, 128).to(device)
    solver = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr)

    duration = []
    frame_count = 0
    lifespan = [[0] for _ in range(num_envs)]
    s_gotten = None

    while frame_count * n_step < max_frame:
        obs_l, acts_l, rews_l, dones_l, probs_l = [], [], [], [], []
        accept_sample = [True for _ in range(num_envs)]
        for _ in range(n_step):
            obs = envs.reset() if s_gotten is None else s_gotten
            obs_in = torch.FloatTensor(obs).to(device)
            prob = actor(obs_in)

            with torch.no_grad():
                a = prob.multinomial(num_samples=1)
            s_gotten, rews, dones, _ = envs.step(a.view(-1).numpy())

            for i in range(num_envs):
                lifespan[i][-1] += 1
                if dones[i]:
                    if lifespan[i][-1] < 500:
                        rews[i] = PENALTY
                    else:  # 500번째
                        accept_sample[i] = False
                        print(lifespan[i][-1], critic(obs_in[[i], :]).view(-1).item())
                    duration.append(lifespan[i][-1])
                    lifespan[i].append(0)

            obs_l.append(obs)
            acts_l.append(a)
            rews_l.append(rews)
            dones_l.append(dones)
            probs_l.append(prob)

        accept_sample = torch.BoolTensor(accept_sample).float().to(device)
        num_sample = accept_sample.sum()
        if num_sample > 0:
            solver.zero_grad()
            obs_l = torch.FloatTensor(obs_l).to(device)
            probs_l = torch.stack(probs_l, dim=0)
            acts_l = torch.stack(acts_l, dim=0)
            probs_a = probs_l.gather(2, acts_l).squeeze(-1)
            next_obs_in = torch.FloatTensor(s_gotten).to(device)
            next_v = critic(next_obs_in).squeeze(-1)
            values = critic(obs_l).squeeze(-1)
            td_res = compute_td(next_v.detach(), rews_l, dones_l, gamma) - values

            critic_loss = td_res.pow(2).sum(dim=0)
            performance = probs_a.log() * td_res.detach()
            actor_loss = -performance.sum(dim=0)
            neg_entropy = (probs_l * probs_l.log()).sum(dim=[0, 2])

            critic_loss = (critic_loss * accept_sample).sum() / num_sample
            actor_loss = (actor_loss * accept_sample).sum() / num_sample
            neg_entropy = (neg_entropy * accept_sample).sum() / num_sample
            loss = critic_loss + actor_loss + 0.5 * neg_entropy
            loss.backward()
            solver.step()

        frame_count += 1
        if frame_count % 101 == 100:
            print(np.array(duration)[-100:].mean())

    envs.close()
    render_simulation(env_name, actor, device)
    plot_duration(duration)


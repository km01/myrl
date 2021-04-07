import gym
from common.multiprocessing_env import SubprocVecEnv, make_env
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from memory import Memory, Transition
from dueling_dqn import Model
import time

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 64
lr = 0.001
initial_exploration = 1000
update_target = 200
replay_memory_capacity = 30000
max_frame = 70000
PENALTY = -1.0
num_envs = 8


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = SubprocVecEnv([make_env(env_name) for i in range(num_envs)])

    net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
    agent = Model(net, 2)
    solver = optim.Adam(agent.parameters())
    memory = Memory(replay_memory_capacity)

    eps = 1.0
    duration = []
    frame_count = 0

    lifespan = [[0] for _ in range(num_envs)]

    s_gotten = None
    while frame_count < max_frame:
        s = envs.reset() if s_gotten is None else s_gotten
        preprocessed_s = torch.FloatTensor(s)
        a = agent.response(preprocessed_s, eps)
        s_gotten, r, done, _ = envs.step(a)

        for i in range(num_envs):
            lifespan[i][-1] += 1
            if done[i]:
                if lifespan[i][-1] < 500:
                    r[i] = PENALTY
                    memory.push(s[i], a[i], r[i], s_gotten[i], done[i])
                duration.append(lifespan[i][-1])
                lifespan[i].append(0)

            if lifespan[i][-1] > 0:  # 500일때 버림.
                memory.push(s[i], a[i], r[i], s_gotten[i], done[i])

        if frame_count > initial_exploration:
            eps -= 0.00005
            eps = max(eps, 0.1)

            batch = memory.sample(batch_size)
            s = torch.FloatTensor([*batch.s]).to(device)
            a = torch.LongTensor([*batch.a]).unsqueeze(-1).to(device)
            r = torch.FloatTensor([*batch.r]).unsqueeze(-1).to(device)
            ns = torch.FloatTensor([*batch.ns]).to(device)
            nt = torch.BoolTensor(np.array(batch.nt).tolist()).unsqueeze(-1).to(device)

            agent.train_model(Transition(s, a, r, ns, nt), solver, gamma, F.mse_loss)

            if frame_count % update_target == 0:
                agent.update()
                if len(duration) > 100:
                    score = np.array(duration)[-100:].mean()
                    print('score:', score)
                    if score > 497:
                        break
        frame_count += 1
    envs.close()

    env = gym.make(env_name)
    s = env.reset()
    while True:
        preprocessed_s = torch.FloatTensor(s).unsqueeze(0).to(device)
        a = agent.response(preprocessed_s)
        s, r, done, _ = env.step(a[0])
        env.render()
        time.sleep(0.02)
        if done:
            break
    env.close()



import torch
import numpy as np
from env_utils import make_vec_env


@torch.no_grad()
def evaluate(env_name, num_env, a2c, det=True):
    env = make_vec_env(env_name, num_env, multi_processing=False)
    obs = env.reset()
    end = [False for _ in range(num_env)]
    life = [0. for _ in range(num_env)]
    gain = [0. for _ in range(num_env)]

    running_h = a2c.initial_state(num_env, requires_grad=False)

    while True:
        env_act, agent_info, running_h = a2c.step(obs, running_h, det)
        obs, rew, done, env_info = env.step(env_act)
        running_h = a2c.masked_fill_initial_state(running_h, done)
        running_h = a2c.masked_fill_initial_state(running_h, end)

        for i in range(num_env):
            if not end[i]:
                life[i] += 1.
                gain[i] += rew[i]
                if done[i]:
                    end[i] = True

        if False not in end:
            break

    life = np.array(life).mean()
    gain = np.array(gain).mean()
    env.close()
    return gain, life

import time
import gym
from common.multiprocessing_env import SubprocVecEnv, make_env
from sac import *
from utils import *
from model import *
from memory import Memory
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    env_name = 'Pendulum-v0'
    num_envs = 8
    envs = [make_env(env_name) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    actor_lr = 2e-4
    critic_lr = 3e-4

    gamma = 0.99
    alpha = 0.3
    soft_tau = 0.001

    sac = SAC(Actor(),
              Critic(),
              gamma,
              alpha,
              soft_tau,
              actor_lr,
              critic_lr)

    memory_size = 500000
    memory = Memory(memory_size)

    evaluation_step = 500
    batch_size = 16 * num_envs
    max_iter = 50000

    frame_count = 0
    test_gains = []

    obs = None
    train_step = 5

    while frame_count < max_iter:

        obs, infos = pendulum_step(obs, envs, sac)
        memory.push(infos)

        if len(memory) >= batch_size:
            for _ in range(train_step):
                batch = memory.sample(batch_size)
                sac.train(batch)

        if frame_count % evaluation_step == 0:
            test_reward_avg = np.mean([test_env(env_name, sac) for _ in range(10)])
            print('[{}/{}]\taverage reward: {}'.format(frame_count, max_iter, test_reward_avg))
            render_simulation(env_name, sac)

        frame_count += 1

from common.multiprocessing_env import SubprocVecEnv, make_env
from sac import SAC, one_step
from memory import Memory
from sac_pendulum import *
import warnings
from rollout_memory import RolloutMemory
from model_pendulum import ModelPendulum
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    env_name = 'Pendulum-v0'
    num_envs = 8
    envs = [make_env(env_name) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    actor_lr = 2e-4
    critic_lr = 3e-4
    model_lr = 5e-4
    gamma = 0.99
    alpha = 0.001
    soft_tau = 0.001

    agent = SAC(Actor(),
                Critic2(),
                gamma,
                alpha,
                soft_tau,
                actor_lr,
                critic_lr)

    model = ModelPendulum(h_size=16, z_size=4, lr=model_lr)
    rollout_mem = RolloutMemory(num_envs, rollout_maxlen=100, max_queue_len=5000)

    memory_size = 500000
    memory = Memory(memory_size)
    eval_freq = 100
    batch_size = 256

    n_steps = 1
    num_sac_iter = 5
    max_iteration = 50000

    frame_count = 0
    test_gains = []
    obs = None

    while frame_count < max_iteration:
        obs, trs = one_step(obs, envs, agent.actor_target)
        memory.push_many(*trs)
        rollout_mem.push(trs[0], trs[1], trs[2], trs[3])

        if len(memory) >= batch_size:
            for _ in range(num_sac_iter):
                batch = memory.sample(batch_size)
                batch = transform_data(batch)
                agent.train(batch)

        if frame_count % 5 == 0 and len(rollout_mem) >= 4:
            print(frame_count, len(rollout_mem))
            world_batch = rollout_mem.fetch_data(16, 50)
            model.train(world_batch)

        if frame_count % eval_freq == eval_freq-1:
            test_reward_avg = np.mean([test_env(env_name, agent.actor) for _ in range(10)])
            print('[{}/{}]\taverage reward: {}'.format(frame_count, max_iteration, test_reward_avg))
            render_simulation(env_name, agent.actor)

        frame_count += 1

    envs.close()
    render_simulation(env_name, agent.actor)

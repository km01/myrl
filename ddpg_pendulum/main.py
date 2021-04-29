from common.multiprocessing_env import SubprocVecEnv, make_env
from ddpg import *
from utils import *
from memory import Memory
from ddpg_pendulum_net import Actor, Critic
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
    soft_tau = 0.001
    model = DDPG(Actor(),
                 Critic(),
                 gamma,
                 soft_tau,
                 actor_lr,
                 critic_lr)

    memory_size = 500000
    memory = Memory(memory_size)
    eval_freq = 1000
    batch_size = 256
    n_steps = 1
    num_ddpg_iter = 5
    max_iteration = 50000

    frame_count = 0
    test_gains = []
    obs = None
    while frame_count < max_iteration:
        obs, trs = n_step(obs, envs, model, n_steps)
        memory.push_n_step_samples(*trs)

        if len(memory) >= batch_size:
            for _ in range(num_ddpg_iter):
                batch = memory.sample(batch_size)
                batch = transform_data(batch)
                model.train(batch)

        if frame_count % eval_freq == 0:
            test_reward_avg = np.mean([test_env(env_name, model) for _ in range(10)])
            print('[{}/{}]\taverage reward: {}'.format(frame_count, max_iteration, test_reward_avg))
            render_simulation(env_name, model)

        frame_count += 1

    envs.close()
    render_simulation(env_name, model)

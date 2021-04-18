from common.multiprocessing_env import SubprocVecEnv, make_env
import torch.optim as optim
from model import *
from utils import *
from memory import Memory
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

env_name = 'Pendulum-v0'
num_envs = 8

n_steps = 32
gamma = 0.9
tau = 0.5

num_samples = 1000
batch_size = 128
ppo_epoch = 10
memory_size = 5000
iteration = 2000

actor_lr = 2e-5
critic_lr = 1e-4
max_grad_norm = 0.5

clip_param = 0.2

actor = Actor()
critic = Critic()
a_solver = optim.Adam(actor.parameters(), lr=actor_lr)
c_solver = optim.Adam(critic.parameters(), lr=critic_lr)


if __name__ == '__main__':
    envs = SubprocVecEnv([make_env(env_name) for i in range(num_envs)])
    frame_count = 0
    test_gains = []
    obs_gotten = None
    memory = Memory(memory_size)
    while frame_count < iteration:

        # sampling
        n = 0
        while num_samples > n:
            obs_gotten, *info = sample_n_step(obs_gotten, actor, envs, n_steps)
            ob_env, act, rew, done, loc, scale = info
            rew = (rew + 8.) / 8.
            gae, val = compute_gae(ob_env, rew, done, obs_gotten, critic, gamma, tau)
            memory.push_n_step_samples(ob_env, act, gae, val, loc, scale)
            n += num_envs * n_steps

        # learning
        a_loss, c_loss = ppo_train(memory,
                                   batch_size,
                                   ppo_epoch,
                                   actor,
                                   critic,
                                   a_solver,
                                   c_solver,
                                   clip_param,
                                   max_grad_norm)

        if frame_count % 10 == 0:
            test_gains.append(np.mean([test_env(env_name, actor) for _ in range(10)]))
            print('[{}/{}]\taverage reward: {}'.format(frame_count, iteration, test_gains[-1]))

        frame_count += 1

    envs.close()
    plot_array(test_gains, 'frame', 'gain')
    render_simulation(env_name, actor)

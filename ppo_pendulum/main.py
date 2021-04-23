from common.multiprocessing_env import SubprocVecEnv, make_env
import torch.optim as optim
from ppo_pendulum import PendulumPPO2
from utils import *
from memory import Memory
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

env_name = 'Pendulum-v0'
num_envs = 8

n_steps = 32
gamma = 0.9
tau = 0.5

num_samples = num_envs * n_steps
batch_size = 128
ppo_epoch = 10
memory_size = 1000
iteration = 2000

max_grad_norm = 0.5
clip_param = 0.2
w_actor = 0.1

w_entropy = w_actor * 0.001
learning_rate = 3e-4
agent = PendulumPPO2()
solver = optim.Adam(agent.parameters(), lr=learning_rate)

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
            ob_env_now, trans, probs = agent.sample_n_step(obs_gotten, envs, n_steps)
            ob_env, act, rew, done = trans
            rew = (rew + 8.) / 8.
            gae, val = agent.compute_gae(ob_env, rew, done, ob_env_now, gamma, tau)
            memory.push_n_step_samples(ob_env, act, gae, val, *probs)
            n += num_envs * n_steps

        # learning
        a_loss, c_loss = agent.ppo_train(memory,
                                         batch_size,
                                         ppo_epoch,
                                         solver,
                                         clip_param,
                                         w_actor,
                                         w_entropy,
                                         max_grad_norm)

        if frame_count % 10 == 0:
            test_gains.append(np.mean([test_env(env_name, agent) for _ in range(10)]))
            print('[{}/{}]\taverage reward: {}'.format(frame_count, iteration, test_gains[-1]))
            render_simulation(env_name, agent)

        frame_count += 1

    envs.close()
    plot_array(test_gains, 'frame', 'gain')
    render_simulation(env_name, agent)

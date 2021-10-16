import torch
from model import ActorCritic
from env_utils import make_vec_env, is_truncated
from rl_utils import get_proper_policy_class, RunningMeanStd
from storage import OnlineRolloutStorage
from evaluate import evaluate
from rppo import Rppo
from logger import Logger
import numpy as np
from rppo_solver import RppoSolver


if __name__ == '__main__':
    # env_name = 'InvertedDoublePendulum-v2'
    # env_name = 'InvertedPendulum-v2'
    # env_name = 'BipedalWalker-v3'

    # env_name = 'LunarLander-v2'
    env_name = 'LunarLanderContinuous-v2'

    # env_name = 'Pendulum-v0'
    # env_name = 'CartPole-v1'

    num_envs = 16
    max_frame = 100000
    n_steps = 8

    cuda = True

    device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

    env = make_vec_env(env_name, num_envs, multi_processing=True)

    tanh_if_continuous = False

    policy_class, action_size, num_policy_param = get_proper_policy_class(env, tanh_if_continuous)

    logger = Logger(num_envs=num_envs)

    obs_size = env.observation_space.shape[0]
    obs_enc_size = 128

    act_size = action_size
    act_enc_size = 16
    policy_param_size = num_policy_param
    policy_class = policy_class

    belief_size = 128

    a2c_input_size = 128
    a2c_hidden_size = 256
    a2c_shared_layers = 3
    a2c_actor_layers = 3
    a2c_critic_layers = 3

    agent = Rppo(obs_size=obs_size,
                 obs_enc_size=obs_enc_size,

                 policy_param_size=policy_param_size,
                 policy_class=policy_class,

                 belief_size=belief_size,
                 a2c_input_size=a2c_input_size,
                 a2c_hidden_size=a2c_hidden_size,
                 a2c_shared_layers=a2c_shared_layers,
                 a2c_actor_layers=a2c_actor_layers,
                 a2c_critic_layers=a2c_critic_layers).to(device)

    lr = 0.0002
    batch_size = 8
    num_batch = 8
    vf_coef = 1.0
    ent_coef = 0.0001
    vf_clip_range = 0.2
    max_grad_norm = 0.5
    clip_range = 0.2
    gamma = 0.99
    lamda = 0.95

    solver = RppoSolver(actor_critic=agent,
                        lr=lr,
                        batch_size=batch_size,
                        num_batch_epoch=num_batch,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        clip_range=clip_range,
                        gamma=gamma,
                        max_grad_norm=max_grad_norm,
                        vf_clip_range=vf_clip_range,
                        lamda=lamda)

    log_interval = 1000
    eval_interval = 10000
    use_reward_rms = True

    if env_name == 'CartPole-v1':
        use_reward_rms = False

    reward_rms = RunningMeanStd() if use_reward_rms else None

    frame_count = 0
    experience = OnlineRolloutStorage(maxlen=n_steps)

    obs = env.reset()

    while frame_count < max_frame:

        running_h = agent.initial_state(num_envs, requires_grad=False)
        first = np.array([True for _ in range(num_envs)])

        for t in range(n_steps):
            running_h = agent.masked_fill_initial_state(running_h, first)
            act, agent_info, running_h = agent.step(obs, running_h)
            obs_next, rew, done, env_info = env.step(act)

            logger.update_env_stats(rew, done)

            transition = {'obs': obs,
                          'hid': agent_info['hid'],
                          'act': agent_info['act'],
                          'val': agent_info['val'],
                          'log_prob': agent_info['log_prob'],
                          'rew': rew,
                          'done': done,
                          'first': first,
                          'truncated': is_truncated(env_info)}

            if reward_rms is not None:
                transition['rew'] = reward_rms.normalize(rew)

            experience.push(transition)

            frame_count += 1
            first = done
            obs = obs_next

            if frame_count % log_interval == 0:
                logger.print_training_stats(100, 5, max_frame)

            if frame_count % eval_interval == 0:
                te_gain, te_life = evaluate(env_name, num_env=10, a2c=agent)
                print('{:-^50}'.format(''))
                print('test | avg gain    : {:.2f}'.format(te_gain))
                print('test | avg lifespan: {:.2f}'.format(te_life))
                print('{:-^50}'.format(''))

        loss_stats = solver.update_calibrate(agent, experience)
        # loss_stats = solver.update(agent, experience)
        logger.update_loss_stats(loss_stats)

    env.close()

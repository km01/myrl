import numpy as np
import torch
from model import ActorCritic
from distributions import TanhGaussian, get_proper_policy_class, Gaussian
from env_utils import make_vec_env, is_truncated, make_test_env, RunningMeanStd, evaluate
from storage import OnlineRolloutStorage
from ppo import PPOSolver
from logger import Logger


if __name__ == '__main__':
    # env_name = 'InvertedDoublePendulum-v2'
    # env_name = 'InvertedPendulum-v2'
    # env_name = 'BipedalWalker-v3'

    # env_name = 'LunarLanderContinuous-v2'
    env_name = 'CartPole-v1'
    # env_name = 'Pendulum-v0'
    num_envs = 16
    max_frame = 200000
    n_steps = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    envs = make_vec_env(env_name, num_envs)
    policy_distribution, policy_param_size = get_proper_policy_class(envs)

    logger = Logger(num_envs=num_envs)

    num_shared_layers = 4
    num_actor_layers = 3
    num_critic_layers = 2
    hidden_size = 256

    actor_critic = ActorCritic(input_size=envs.observation_space.shape[0],
                               action_param_size=policy_param_size,
                               hidden_size=hidden_size,
                               num_shared_layers=num_shared_layers,
                               num_actor_layers=num_actor_layers,
                               num_critic_layers=num_critic_layers,
                               policy_distribution=policy_distribution,
                               device=device)

    use_reward_rms = False if env_name == 'CartPole-v1' else True
    reward_rms = RunningMeanStd(use_reward_rms)

    lr = 0.0001
    batch_size = 256
    num_batch = 4
    vf_coef = 1.0
    ent_coef = 0.1
    vf_clip_range = 0.2
    max_grad_norm = 0.5
    clip_range = 0.2
    gamma = 0.99
    lamda = 0.95

    log_interval = 1000
    eval_interval = 10000

    solver = PPOSolver(actor_critic=actor_critic,
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

    frame_count = 0
    experience = OnlineRolloutStorage(maxlen=n_steps)
    obs = envs.reset()

    while frame_count < max_frame:

        action, agent_info = actor_critic.select_action(obs)
        obs_next, rews, dones, env_info = envs.step(action)
        logger.update_env_stats(rews, dones)
        reward_rms.update(rews)
        experience.push({'obs': obs,
                         'act': agent_info['act'],
                         'val': agent_info['val'],
                         'log_prob': agent_info['log_prob'],
                         'rew': reward_rms.normalize(rews),
                         'done': dones,
                         'truncated': is_truncated(env_info)})
        obs, frame_count = obs_next, frame_count + 1

        if frame_count % n_steps == n_steps - 1:
            loss_stats = solver.update(actor_critic, experience)
            logger.update_loss_stats(loss_stats)

        if frame_count % log_interval == 0:
            logger.print_training_stats(100, 5, max_frame)

        if frame_count % eval_interval == 0:
            te_gain, te_life = evaluate(env_name, actor_critic)
            print('{:-^50}'.format(''))
            print('test | avg gain    : {:.2f}'.format(te_gain))
            print('test | avg lifespan: {:.2f}'.format(te_life))
            print('{:-^50}'.format(''))

    envs.close()

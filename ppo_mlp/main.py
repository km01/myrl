import torch
from model import ActorCritic
from env_utils import make_vec_env, is_truncated
from rl_utils import get_proper_policy_class, evaluate, RunningMeanStd
from storage import OnlineRolloutStorage
from ppo import PPOSolver
from logger import Logger

if __name__ == '__main__':
    # env_name = 'InvertedDoublePendulum-v2'
    # env_name = 'InvertedPendulum-v2'
    # env_name = 'BipedalWalker-v3'
    env_name = 'LunarLanderContinuous-v2'
    # env_name = 'Pendulum-v0'
    # env_name = 'CartPole-v1'
    num_envs = 16
    max_frame = 1000000
    n_steps = 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    envs = make_vec_env(env_name, num_envs)

    tanh_if_continuous = True

    policy_class, policy_param_size = get_proper_policy_class(envs, tanh_if_continuous)

    logger = Logger(num_envs=num_envs)

    num_shared_layers = 3
    num_actor_layers = 3
    num_critic_layers = 4
    hidden_size = 256

    actor_critic = ActorCritic(input_size=envs.observation_space.shape[0],
                               action_param_size=policy_param_size,
                               hidden_size=hidden_size,
                               policy_class=policy_class,
                               num_actor_layers=num_actor_layers,
                               num_critic_layers=num_critic_layers,
                               num_shared_layers=num_shared_layers).to(device)

    lr = 0.00001
    batch_size = 256
    num_batch = 8
    vf_coef = 1.0
    ent_coef = 0.001
    vf_clip_range = 0.2
    max_grad_norm = 0.5
    clip_range = 0.2
    gamma = 0.99
    lamda = 0.95

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

    log_interval = 1000
    eval_interval = 10000
    use_reward_rms = True

    if env_name == 'CartPole-v1':
        use_reward_rms = False

    reward_rms = RunningMeanStd() if use_reward_rms else None

    frame_count = 0
    experience = OnlineRolloutStorage(maxlen=n_steps)
    obs = envs.reset()

    while frame_count < max_frame:

        action, agent_info = actor_critic.select_action(obs)
        obs_next, rews, dones, env_info = envs.step(action)

        transition = {'obs': obs,
                      'act': agent_info['act'],
                      'val': agent_info['val'],
                      'log_prob': agent_info['log_prob'],
                      'rew': rews,
                      'done': dones,
                      'truncated': is_truncated(env_info)}

        if reward_rms is not None:
            reward_rms.update(rews)
            transition['rew'] = reward_rms.normalize(rews)

        experience.push(transition)
        logger.update_env_stats(rews, dones)
        obs, frame_count = obs_next, frame_count + 1

        if frame_count % n_steps == 0:
            loss_stats = solver.update(actor_critic, experience)
            logger.update_loss_stats(loss_stats)

        if frame_count % log_interval == 0:
            logger.print_training_stats(100, 5, max_frame)

        if frame_count % eval_interval == 0:
            te_gain, te_life = evaluate(env_name, num_env=10, actor_critic=actor_critic)
            print('{:-^50}'.format(''))
            print('test | avg gain    : {:.2f}'.format(te_gain))
            print('test | avg lifespan: {:.2f}'.format(te_life))
            print('{:-^50}'.format(''))

    envs.close()

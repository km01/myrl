import argparse
import gym
import itertools
from utils import *
from sac import SAC, SACV
from replay_memory import ReplayMemory


parser = argparse.ArgumentParser()

parser.add_argument('--env-name',               default="Pendulum-v0")
parser.add_argument('--eval',                   default=True)
parser.add_argument('--gamma',                  default=0.99)
parser.add_argument('--tau',                    default=0.001)

parser.add_argument('--use_value_net',            default=False)

parser.add_argument('--lr',                     default=0.0001)
parser.add_argument('--alpha',                  default=0.2)
parser.add_argument('--hidden_size',            default=128)

parser.add_argument('--seed',                   default=123456)
parser.add_argument('--batch_size',             default=256)

parser.add_argument('--replay_size',            default=1000000,
                    help='size of replay buffer')

parser.add_argument('--max_steps',              default=1000000,
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--num_initial_steps',      default=10000,
                    help='steps sampling random actions')

parser.add_argument('--updates_per_step',       default=1)
parser.add_argument('--eval_per_episode',       default=10)
parser.add_argument('--num_eval_episode',       default=10)

parser.add_argument('--target_update_interval', default=1,
                    help='Value target update per no. of updates per step')

parser.add_argument('--cuda',                   default=True)
args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)
set_seed_everywhere(args.seed)

sac = SACV if args.use_value_net else SAC


model = sac(input_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            hidden_size=args.hidden_size,
            lr=args.lr,
            device=torch.device('cuda' if args.cuda else 'cpu'))

memory = ReplayMemory(args.replay_size)

total_steps = 0

sample_episode(memory, env, args.num_initial_steps)

for i_episode in itertools.count(1):
    gain, life_span, loss_info = run_episode(model, env, memory, args.batch_size, args.updates_per_step)
    total_steps += life_span
    print("Episode: {}, total steps: {}, life span: {}, reward: {}"
          .format(i_episode, total_steps, life_span, round(gain, 2)))

    print(loss_info)

    if args.eval is True and i_episode % args.eval_per_episode == 0:
        eval_gain, eval_life_span = evaluate_episode(model, env, args.num_eval_episode)
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. reward: {}, lifespan: {}"
              .format(args.num_eval_episode, round(eval_gain, 2), int(eval_life_span)))
        print("----------------------------------------")

    if i_episode >= args.max_steps:
        break

env.close()

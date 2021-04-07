import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('s', 'a', 'r', 'ns', 'nt'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, s, a, r, ns, nt):
        self.memory.append(Transition(s, a, r, ns, nt))

    def push_many(self, s_list, a_list, r_list, ns_list, nt_list):
        for s, a, r, ns, nt in zip(s_list, a_list, r_list, ns_list, nt_list):
            self.push(s, a, r, ns, nt)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)



# if __name__ == '_main__':
#
#     env_name = 'CartPole-v1'
#     gamma = 0.99
#     batch_size = 32
#     lr = 0.001
#     initial_exploration = 1000
#     update_target = 100
#     replay_memory_capacity = 50000
#     max_frame = 3000
#     PENALTY = -1.0
#     num_envs = 8
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     envs = SubprocVecEnv([make_env(env_name) for i in range(num_envs)])
#     net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
#     agent = Model(net, 2)
#     solver = optim.Adam(agent.parameters())
#     memory = Memory(replay_memory_capacity)
#
#     eps = 1.0
#     duration = []
#     frame_count = 0
#
#     lifespan = [[0] for _ in range(num_envs)]
#     s_gotten = None
#     while frame_count < max_frame:
#         s = envs.reset() if s_gotten is None else s_gotten
#         preprocessed_s = torch.FloatTensor(s)
#         a = agent.response(preprocessed_s, eps)
#         s_gotten, r, done, _ = envs.step(a)
#
#         for i in range(num_envs):
#             lifespan[i][-1] += 1
#             if done[i]:
#                 if lifespan[i][-1] < 500:
#                     r[i] = PENALTY
#                 duration.append(lifespan[i][-1])
#                 print(duration[-1])
#                 lifespan[i].append(0)
#
#             memory.push(s[i], a[i], r[i], s_gotten[i], done[i])
#
#         if frame_count > initial_exploration:
#             eps -= 0.00005
#             eps = max(eps, 0.1)
#
#             batch = memory.sample(batch_size)
#             s = torch.FloatTensor([*batch.s])
#             a = torch.LongTensor([*batch.a]).unsqueeze(-1)
#             r = torch.FloatTensor([*batch.r]).unsqueeze(-1)
#             ns = torch.FloatTensor([*batch.ns])
#             nt = torch.BoolTensor([*batch.nt]).unsqueeze(-1)
#
#             agent.train_model(Transition(s, a, r, ns, nt), solver, gamma, F.mse_loss)
#
#             if frame_count % update_target == 0:
#                 agent.update()
#
#         frame_count += 1
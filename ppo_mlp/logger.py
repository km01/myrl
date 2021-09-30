import numpy as np


class Logger(object):

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.gain_buffer = [0. for _ in range(num_envs)]
        self.life_span_buffer = [0 for _ in range(num_envs)]

        self.gain_recoder = []
        self.life_span_recoder = []
        self.loss_recoder = {}
        self.step_count = 0
        self.num_episode = 0

    def update_env_stats(self, rew, done):
        for i in range(self.num_envs):
            self.gain_buffer[i] += rew[i]
            self.life_span_buffer[i] += 1

            if done[i]:
                self.gain_recoder.append(self.gain_buffer[i])
                self.life_span_recoder.append(self.life_span_buffer[i])
                self.gain_buffer[i] = 0.
                self.life_span_buffer[i] = 0
                self.num_episode += 1

        self.step_count += 1

    def update_loss_stats(self, loss_info):
        for key, val in loss_info.items():
            if key not in self.loss_recoder.keys():
                self.loss_recoder[key] = []

            self.loss_recoder[key].append(val)

    def print_training_stats(self, num_epi_sample, num_loss_sample, max_frame):

        num_epi_sample = min(len(self.gain_recoder), num_epi_sample)

        if num_epi_sample == 0:
            avg_gain = 0.
            avg_life = 0
        else:
            avg_gain = float(np.array(self.gain_recoder[-num_epi_sample:], dtype=np.float).mean())
            avg_life = float(np.array(self.life_span_recoder[-num_epi_sample:], dtype=np.float).mean())

        print('{:-^50}'.format('[' + str(self.step_count) + '/' + str(max_frame) + ']'))

        print('total episode       : {}'.format(self.num_episode))
        print('total step          : {}'.format(self.step_count * self.num_envs))
        print('avg training gain   : {:.2f}'.format(avg_gain))
        print('avg training epi_len: {:.2f}'.format(avg_life))
        for key, val in self.loss_recoder.items():
            length = min(len(val), num_loss_sample)
            avg = float(np.array(val[-length:], dtype=np.float).mean())
            print('{:<20}: {:.7f}'.format(key, avg))

        print('{:-^50}'.format(''))

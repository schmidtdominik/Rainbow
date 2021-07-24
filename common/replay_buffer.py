import collections
import random
from math import sqrt

import numpy as np
import torch
from gym.wrappers import LazyFrames

from common.utils import prep_observation_for_qnet


class UniformReplayBuffer:
    def __init__(self, burnin, capacity, gamma, n_step, parallel_envs, use_amp):
        self.capacity = capacity
        self.burnin = burnin
        self.buffer = []
        self.nextwrite = 0
        self.use_amp = use_amp

        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffers = [collections.deque(maxlen=self.n_step + 1) for j in range(parallel_envs)]

    def put(self, *transition, j):
        self.n_step_buffers[j].append(transition)
        if len(self.n_step_buffers[j]) == self.n_step + 1 and not self.n_step_buffers[j][0][3]:  # n-step transition can't start with terminal state
            state = self.n_step_buffers[j][0][0]
            action = self.n_step_buffers[j][0][1]
            next_state = self.n_step_buffers[j][self.n_step][0]
            done = self.n_step_buffers[j][self.n_step][3]
            reward = self.n_step_buffers[j][0][2]
            for k in range(1, self.n_step):
                reward += self.n_step_buffers[j][k][2] * self.gamma ** k
                if self.n_step_buffers[j][k][3]:
                    done = True
                    break

            action = torch.LongTensor([action]).cuda()
            reward = torch.FloatTensor([reward]).cuda()
            done = torch.FloatTensor([done]).cuda()

            if len(self.buffer) < self.capacity:
                self.buffer.append((state, next_state, action, reward, done))
            else:
                self.buffer[self.nextwrite % self.capacity] = (state, next_state, action, reward, done)
                self.nextwrite += 1

    def sample(self, batch_size, beta=None):
        """ Sample a minibatch from the ER buffer (also converts the FrameStacked LazyFrames to contiguous tensors) """
        batch = random.sample(self.buffer, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        state = list(map(lambda x: torch.from_numpy(x.__array__()), state))
        next_state = list(map(lambda x: torch.from_numpy(x.__array__()), next_state))

        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        return prep_observation_for_qnet(state, self.use_amp), prep_observation_for_qnet(next_state, self.use_amp), \
               action.squeeze(), reward.squeeze(), done.squeeze()

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """ based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments,
    removed alpha hyperparameter like google/dopamine
    """

    def __init__(self, burnin: int, capacity: int, gamma: float, n_step: int, parallel_envs: int, use_amp):
        self.burnin = burnin
        self.capacity = capacity  # must be a power of two
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffers = [collections.deque(maxlen=self.n_step + 1) for j in range(parallel_envs)]

        self.use_amp = use_amp

        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.max_priority = 1.0  # initial priority of new transitions

        self.data = [None for _ in range(self.capacity)]  # cyclical buffer for transitions
        self.next_idx = 0  # next write location
        self.size = 0  # number of buffer elements

    @staticmethod
    def prepare_transition(state, next_state, action: int, reward: float, done: bool):
        action = torch.LongTensor([action]).cuda()
        reward = torch.FloatTensor([reward]).cuda()
        done = torch.FloatTensor([done]).cuda()

        return state, next_state, action, reward, done

    def put(self, *transition, j):
        self.n_step_buffers[j].append(transition)
        if len(self.n_step_buffers[j]) == self.n_step + 1 and not self.n_step_buffers[j][0][3]:  # n-step transition can't start with terminal state
            state = self.n_step_buffers[j][0][0]
            action = self.n_step_buffers[j][0][1]
            next_state = self.n_step_buffers[j][self.n_step][0]
            done = self.n_step_buffers[j][self.n_step][3]
            reward = self.n_step_buffers[j][0][2]
            for k in range(1, self.n_step):
                reward += self.n_step_buffers[j][k][2] * self.gamma ** k
                if self.n_step_buffers[j][k][3]:
                    done = True
                    break

            assert isinstance(state, LazyFrames)
            assert isinstance(next_state, LazyFrames)

            idx = self.next_idx
            self.data[idx] = self.prepare_transition(state, next_state, action, reward, done)
            self.next_idx = (idx + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

            self._set_priority_min(idx, sqrt(self.max_priority))
            self._set_priority_sum(idx, sqrt(self.max_priority))

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """ find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight

        samples = []
        for i in indices:
            samples.append(self.data[i])

        return indices, weights, self.prepare_samples(samples)

    def prepare_samples(self, batch):
        state, next_state, action, reward, done = zip(*batch)
        state = list(map(lambda x: torch.from_numpy(x.__array__()), state))
        next_state = list(map(lambda x: torch.from_numpy(x.__array__()), next_state))

        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        return prep_observation_for_qnet(state, self.use_amp), prep_observation_for_qnet(next_state, self.use_amp), \
               action.squeeze(), reward.squeeze(), done.squeeze()

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    @property
    def is_full(self):
        return self.capacity == self.size

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return self.size


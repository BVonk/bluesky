from collections import deque
import numpy as np
import random

class ReplayMemory():

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.num_experiences = 0
        self.memory = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.memory, self.num_experiences)
        else:
            return random.sample(self.memory, batch_size)

    def size(self):
        return self.memory_size

    def add(self, state, action, reward, new_state, done, mask=None):
        experience = (state, action, reward, new_state, done, mask)
        if self.num_experiences < self.memory_size:
            self.memory.append(experience)
            self.num_experiences += 1
        else:
            self.memory.popleft()
            self.memory.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.memory = deque()
        self.num_experiences = 0

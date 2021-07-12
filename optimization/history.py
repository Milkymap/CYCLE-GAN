import torch as th 
from torch.autograd import Variable
import numpy as np 

class History:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for matrix in data.data:
            matrix = matrix[None, ...]
            if len(self.data) < self.max_size:
                self.data.append(matrix)
                to_return.append(matrix)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = matrix
                else:
                    to_return.append(matrix)
        return Variable(th.cat(to_return))

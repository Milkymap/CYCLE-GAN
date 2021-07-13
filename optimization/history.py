import torch as th 
from torch.autograd import Variable
import numpy as np 

class History:
	def __init__(self, max_size=50):
		self.max_size = max_size
		self.data = []

	def push_and_pop(self, data):
		to_return = []
		for element in data.data:
			element = th.unsqueeze(element, 0)
			if len(self.data) < self.max_size:
				self.data.append(element)
				to_return.append(element)
			else:
				if np.random.uniform(0, 1) > 0.5:
					i = np.random.randint(0, self.max_size - 1)
					to_return.append(self.data[i].clone())
					self.data[i] = element
				else:
					to_return.append(element)
		return Variable(th.cat(to_return))


import torch as th 
import torch.nn as nn 

import numpy as np 

class CK(nn.Module):
	def __init__(self, i_channels, o_channels, normalize=False):
		super(CK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(i_channels, o_channels, 4, 2, 1),
			nn.Identity() if not normalize else nn.InstanceNorm2d(o_channels),
			nn.LeakyReLU(0.2)
		)

	def forward(self, X):
		return self.body(X)

class Discriminator(nn.Module):
	def __init__(self, i_shape, i_channels, o_channels, nb_blocks):
		super(Discriminator, self).__init__()
		self.last_shape = tuple(np.asarray(i_shape) // 2 ** nb_blocks)
		self.head = CK(i_channels, o_channels)
		self.body = nn.Sequential(*[ 
			CK(o_channels * 2 ** idx, o_channels * 2 ** (idx + 1), True) 
			for idx in range(nb_blocks - 1) 
			]
		)
		self.tail = nn.Conv2d(o_channels * 2 ** (nb_blocks - 1), 1, 1)
		self.apply(self.__initialize_weights)

	def forward(self, X):
		return th.squeeze(self.tail(self.body(self.head(X))))

	def __initialize_weights(self, mdl):
		c_name = mdl.__class__.__name__
		index = c_name.find('Conv')
		if index != -1:
			nn.init.normal_(mdl.weight.data, 0.0, 0.02)
			if hasattr(mdl, 'bias') and mdl.bias is not None:
				nn.init.constant_(mdl.bias.data, 0.0)

if __name__ == '__main__':
	D = Discriminator((128, 128), 3, 64, 4)
	X = th.randn((2, 3, 128, 128))
	
	print(D)
	print(D(X).shape)
	print(D.last_shape)

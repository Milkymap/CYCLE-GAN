import torch as th 
import torch.nn as nn 

class C7S1K(nn.Module):
	def __init__(self, i_channels, o_channels):
		super(C7S1K, self).__init__()
		self.body = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(i_channels, o_channels, 7, 1),
			nn.InstanceNorm2d(o_channels),
			nn.ReLU()
		)

	def forward(self, X):
		return self.body(X)


class DK(nn.Module):
	def __init__(self, i_channels, o_channels):
		super(DK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(i_channels, o_channels, 3, 2, 1),
			nn.InstanceNorm2d(o_channels),
			nn.ReLU()
		)

	def forward(self, X):
		return self.body(X)


class RK(nn.Module):
	def __init__(self, n_filters):
		super(RK, self).__init__()
		self.body = nn.Sequential(
			nn.ReflectionPad2d(1), 
			nn.Conv2d(n_filters, n_filters, 3, 1),
			nn.InstanceNorm2d(n_filters),
			nn.ReLU(),
			nn.ReflectionPad2d(1), 
			nn.Conv2d(n_filters, n_filters, 3, 1),
			nn.InstanceNorm2d(n_filters),
		)

	def forward(self, X):
		return X + self.body(X)


class UK(nn.Module):
	def __init__(self, i_channels, o_channels):
		super(UK, self).__init__()
		self.body = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(i_channels, o_channels, 3, 1, 1), 
			nn.InstanceNorm2d(o_channels),
			nn.ReLU()
		)

	def forward(self, X):
		return self.body(X)


class Generator(nn.Module):
	def __init__(self, i_channels, o_channels, nb_down, nb_rblocks):
		super(Generator, self).__init__()
		self.head = C7S1K(i_channels, o_channels)
		self.down = nn.Sequential(*[ 
				DK(o_channels * 2 ** idx, o_channels * 2 ** (idx + 1)) 
				for idx in range(nb_down) 
			]
		)
		self.body = nn.Sequential(*[ 
				RK(o_channels * 2 ** nb_down) 
				for idx in range(nb_rblocks) 
			]
		)
		self.tail = nn.Sequential(*[ 
				UK(o_channels * 2 ** idx, o_channels * 2 ** (idx - 1)) 
				for idx in range(nb_down, 0, -1) 
			]
		)
		self.term = C7S1K(o_channels, i_channels)
		self.apply(self.__initialize_weights)

	def forward(self, X):	
		return self.term(self.tail(self.body(self.down(self.head(X)))))

	def __initialize_weights(self, mdl):
		c_name = mdl.__class__.__name__
		index = c_name.find('Conv')
		if index != -1:
			nn.init.normal_(mdl.weight.data, 0.0, 0.02)
			if hasattr(mdl, 'bias') and mdl.bias is not None:
				nn.init.constant_(mdl.bias.data, 0.0)


if __name__ == '__main__':
	G = Generator(3, 64, 2, 6)
	X = th.randn(2, 3, 256, 256)
	
	print(G)
	print(G(X).shape)
import numpy as np
import torch as th 
import torch.utils.data as D 
 

from PIL import Image 
from glob import glob 
from os import path 

class DataHolder(D.Dataset):
	def __init__(self, root_for_domain_A, root_for_domain_B, mapper, paired=True):
		assert mapper is not None, 'please, define an image_transformer!' 
		self.domain_A = glob(path.join(root_for_domain_A, '*.jpg'))
		self.domain_B = glob(path.join(root_for_domain_B, '*.jpg'))
		self.mapper = mapper 
		self.paired = paired

	def __len__(self):
		if self.paired:
			return len(self.domain_A)
		return max(len(self.domain_A), len(self.domain_B))

	def __getitem__(self, idx):
		IA = idx % len(self.domain_A) 
		IB = IA if self.paired else np.random.randint(len(self.domain_B))
		XA = Image.open(self.domain_A[IA])
		XB = Image.open(self.domain_B[IB])
		XA = self.mapper(XA)
		XB = self.mapper(XB)
		return XA, XB 


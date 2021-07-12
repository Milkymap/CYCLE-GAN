import click 
import itertools as it 

import torch as th 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as D 

import torchvision.transforms as T 

from optimization.dataholder import DataHolder 
from optimization.history import History
from models.generator import Generator
from models.discriminator import Discriminator

from libraries.log import logger 
from libraries.strategies import * 


@click.command()
@click.option('--gpu_idx', help='index of gpu core', type=int)
@click.option('--height', help='input height for the model', type=int)
@click.option('--width', help='input width for the model', type=int)
@click.option('--source_x0', help='path to X0', type=click.Path(True))
@click.option('--source_x1', help='path to X1', type=click.Path(True))
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--paired/--no-paired', help='is paired or not', default=True)
@click.option('--storage', help='storage dir for sampled data', type=click.Path(True))
def train(gpu_idx, height, width, source_x0, source_x1, nb_epochs, bt_size, paired, storage):
	device = th.device(f'cuda:{gpu_idx}' if th.cuda.is_available() else 'cpu')
	G_A2B = Generator(3, 64, 2, 6).to(device)
	G_B2A = Generator(3, 64, 2, 6).to(device)
	DIS_A = Discriminator((height, width), 3, 64, 4).to(device)
	DIS_B = Discriminator((height, width), 3, 64, 4).to(device)

	IDT_Loss = nn.L1Loss().to(device)
	CYC_Loss = nn.L1Loss().to(device)
	GAN_Loss = nn.MSELoss().to(device)

	LAMBDA_IDT = 5
	LAMBDA_CYC = 10
	LAMBDA_GAN = 1

	OPT_2G = optim.Adam(it.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
	OPT_DA = optim.Adam(DIS_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
	OPT_DB = optim.Adam(DIS_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

	XA_BUFFER = History()
	XB_BUFFER = History()

	mapper = T.Compose([
		T.Resize((height, width)), 
		T.ToTensor(),
		T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

	data_holder = DataHolder(source_x0, source_x1, mapper, paired)
	data_loader = D.DataLoader(dataset=data_holder, shuffle=True, batch_size=bt_size)

	epoch_counter = 0 
	while epoch_counter < nb_epochs:
		for idx, (X_A, X_B) in enumerate(data_loader):
			X_A = X_A.to(device)
			X_B = X_B.to(device)

			RL = th.ones((X_A.shape[0], DIS_A.last_shape[0], DIS_A.last_shape[1])).to(device)
			FL = th.zeros((X_A.shape[0], DIS_A.last_shape[0], DIS_A.last_shape[1])).to(device)

			OPT_2G.zero_grad()
			X_B_ = G_A2B(X_A)
			X_A_ = G_B2A(X_B)

			LC0 = CYC_Loss(G_B2A(X_B_), X_A)  # G_B2A(G_A2B(XA)) ~ XA 
			LC1 = CYC_Loss(G_A2B(X_A_), X_B)  # G_A2B(G_B2A(XB)) ~ XB 
			LC2 = (LC0 + LC1) / 2
			LI0 = IDT_Loss(G_A2B(X_B), X_B)   # G_A2B(XB) ~ XB
			LI1 = IDT_Loss(G_B2A(X_A), X_A)   # G_B2A(XA) ~ XA
			LI2 = (LI0 + LI1) / 2
			LG0 = GAN_Loss(DIS_A(X_A_), RL)   # log[D_A(G_B2A(XB))]
			LG1 = GAN_Loss(DIS_B(X_B_), RL)   # log[D_B(G_A2B(XA))]
			LG2 = (LG0 + LG1) / 2

			TOT = LAMBDA_GAN * LG2 + LAMBDA_CYC * LC2 + LAMBDA_IDT * LI2  

			TOT.backward()
			OPT_2G.step()

			OPT_DA.zero_grad()
			X_AH = XA_BUFFER.push_and_pop(X_A_)
			LDA = (GAN_Loss(DIS_A(X_A), RL) + GAN_Loss(DIS_A(X_AH.detach()), FL)) / 2
			LDA.backward()
			OPT_DA.step()


			OPT_DB.zero_grad()
			X_BH = XB_BUFFER.push_and_pop(X_B_)
			LDB = (GAN_Loss(DIS_A(X_B), RL) + GAN_Loss(DIS_A(X_BH.detach()), FL)) / 2
			LDB.backward()
			OPT_DA.step()

			logger.debug(f'[{nb_epochs:03d}/{epoch_counter:03d}]:{idx:05d} >> TOT : {TOT.item():07.3f}, LDA : {LDA.item():07.3f}, LDB : {LDB.item():07.3f}')
			if idx % 10 == 0:
				X_A = X_A.cpu()
				X_B_ = X_B_.cpu()
				XA_RES = th.cat([X_A, B_A_], dim=-1)
				XA_RES = to_grid(XA_RES, nb_rows=1) 
				XA_RES = th2cv(XA_RES) * 255
				cv2.imwrite(path.join(storage, f'img_{epoch_counter:03d}{idx:03d}.jpg'), XA_RES)

		epoch_counter = epoch_counter + 1

if __name__ == '__main__':
	train()
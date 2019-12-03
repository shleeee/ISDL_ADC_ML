import numpy as np
import torch
import data_io
from torch.autograd import Variable
import util


def network(input_cell,hidden_cell,output_cell):

	model = torch.nn.Sequential(
		torch.nn.Linear(input_cell,hidden_cell),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_cell,output_cell),
		torch.nn.ReLU()
	)

	return model

def network2(input_cell,hidden_cell_1,hidden_cell_2,output_cell):

	model = torch.nn.Sequential(
		torch.nn.Linear(input_cell,hidden_cell_1),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_cell_1,hidden_cell_2),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_cell_2,output_cell),
		torch.nn.ReLU()
	)

	return model

def train(x_train, y_train, network, learning_rate, epochs):

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()		#Loss function

	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		loss_var = loss_fn(y_hat, y_train)
		loss[ix] = loss_var.data.cpu()
		network.zero_grad()
		loss_var.backward()
		
		for param in network.parameters():
			param.data = param.data - learning_rate * param.grad.data

#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )

	return network, loss

def test(x_test, y_test, network):

	Y_hat = network(x_test)
	y_tmp = torch.max(Y_hat, dim=1)[1] #return max index
	y_tmp = y_tmp.data.cpu().numpy()
	acc = np.mean(1 * (y_tmp == y_test))

	return acc

def quantization_train(x_train, y_train, network, learning_rate, epochs, int_bit, float_bit) :

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()
	
	for param in network.parameters():
		param.data = util.quantize(param.data,int_bit,float_bit)

#	network = util.normalize(network)	
	
	
	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		loss_var = loss_fn(y_hat, y_train)
		loss[ix] = loss_var.data.cpu()
		network.zero_grad()
		loss_var.backward()
		
		for param in network.parameters():
			param.data = param.data - util.quantize(learning_rate * param.grad.data,int_bit,float_bit)

#		network = util.normalize(network)
		
		for param in network.parameters():
			param.data = util.quantize(param.data,int_bit,float_bit)
	
#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )

	return network, loss

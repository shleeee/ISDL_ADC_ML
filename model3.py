import numpy as np
import torch
import data_io
from torch.autograd import Variable
import util

k = 0

class cReLU(torch.nn.Module):
	def __init__(self):
		super(cReLU,self).__init__()
	
	def forward(self, input):
		global k
		if(k == 1):
			input = torch.round(input*(2**1))/(2**1)
			output = torch.clamp(input, min=0, max=15.5)
		else:
			output = torch.clamp(input, min=0, max=8)
		return output

def network(input_cell,hidden_cell,output_cell):

	model = torch.nn.Sequential(
		torch.nn.Linear(input_cell,hidden_cell),
		cReLU(),
		torch.nn.Linear(hidden_cell,output_cell),
		cReLU()
	)

	return model

def network2(input_cell,hidden_cell_1,hidden_cell_2,output_cell):

	model = torch.nn.Sequential(
		torch.nn.Linear(input_cell,hidden_cell_1),
		cReLU(),
		torch.nn.Linear(hidden_cell_1,hidden_cell_2),
		cReLU(),
		torch.nn.Linear(hidden_cell_2,output_cell),
		cReLU()
	)

	return model

def network_wo_relu(input_cell,hidden_cell,output_cell):
	model = torch.nn.Sequential(
		torch.nn.Linear(input_cell,hidden_cell),
		torch.nn.Linear(hidden_cell,output_cell)
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

def train2(x_train, y_train, x_train2, y_train2, network, learning_rate, epochs):

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()		#Loss function

	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		y_hat2 = network(x_train2)	#Forward pass
		loss_var1 = loss_fn(y_hat, y_train)
		loss_var2 = loss_fn(y_hat2, y_train2)
		loss_var = (loss_var1+loss_var2)/2.0
		loss[ix] = loss_var.data.cpu()
		network.zero_grad()
		loss_var.backward()
		
		for param in network.parameters():
			param.data = param.data - learning_rate * param.grad.data

#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )

	return network, loss

def train3(x_train, y_train, x_train2, y_train2, x_train3, y_train3, network, learning_rate, epochs):

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()		#Loss function

	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		y_hat2 = network(x_train2)	#Forward pass
		y_hat3 = network(x_train3)	#Forward pass
		loss_var1 = loss_fn(y_hat, y_train)
		loss_var2 = loss_fn(y_hat2, y_train2)
		loss_var3 = loss_fn(y_hat3, y_train3)
		loss_var = (loss_var1+loss_var2+loss_var3)/3.0
		loss[ix] = loss_var.data.cpu()
		network.zero_grad()
		loss_var.backward()
		
		for param in network.parameters():
			param.data = param.data - learning_rate * param.grad.data

#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )

	return network, loss

def train4(x_train, y_train, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, network, learning_rate, epochs):

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()		#Loss function

	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		y_hat2 = network(x_train2)	#Forward pass
		y_hat3 = network(x_train3)	#Forward pass
		y_hat4 = network(x_train4)	#Forward pass
		loss_var1 = loss_fn(y_hat, y_train)
		loss_var2 = loss_fn(y_hat2, y_train2)
		loss_var3 = loss_fn(y_hat3, y_train3)
		loss_var4 = loss_fn(y_hat4, y_train4)
		loss_var = (loss_var1+loss_var2+loss_var3+loss_var4)/4.0
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


def quantization_train2(x_train, y_train, network, learning_rate, epochs, int_bit, float_bit) :

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()
	
	for param in network.parameters():
		param.data = util.quantize(param.data,int_bit,float_bit)


def quantization_train3(x_train, y_train, x_train2, y_train2, x_train3, y_train3, network, learning_rate, epochs, int_bit, float_bit) :

	loss = np.zeros([epochs,1])
	loss_fn = torch.nn.MSELoss()
	
	for param in network.parameters():
		param.data = util.quantize(param.data,int_bit,float_bit)

#	network = util.normalize(network)	
	
	
	for ix in range(epochs):
		y_hat = network(x_train)	#Forward pass
		y_hat2 = network(x_train2)	#Forward pass
		y_hat3 = network(x_train3)	#Forward pass
		loss_var1 = loss_fn(y_hat, y_train)
		loss_var2 = loss_fn(y_hat2, y_train2)
		loss_var3 = loss_fn(y_hat3, y_train3)
		loss_var = (loss_var1+loss_var2+loss_var3)/3.0
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

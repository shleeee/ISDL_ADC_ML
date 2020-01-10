import numpy as np
import torch
import data_io as io
from torch.autograd import Variable
import util

class cReLU(torch.nn.Module):
    def __init__(self):
        super(cReLU,self).__init__()

    def forward(self, input):
        return torch.clamp(input, min=0., max =4.) 

class FFClassifier(torch.nn.Module):
    def __init__(self, input_cell, hidden_cell_1, hidden_cell_2, output_cell):
        super(FFClassifier,self).__init__()
        self.fc1 = torch.nn.Linear(input_cell, hidden_cell_1)
        self.fc2 = torch.nn.Linear(hidden_cell_1, hidden_cell_2)
        self.fc3 = torch.nn.Linear(input_cell, hidden_cell_1)
        self.fc4 = torch.nn.Linear(hidden_cell_1+hidden_cell_2, output_cell)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        y = x.cpu().numpy().copy()
#		l = torch.FloatTensor(k)
        dtype = torch.cuda.FloatTensor
        y = torch.from_numpy(y).type(dtype)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        y = self.fc3(y)
        y = self.relu(y)
#		print('x shape : ', x.shape)
#		print('l shape : ', l.shape)
#		y = x.cpu().numpy().copy()
#		m = np.concatenate((y,k), axis=1)
#		x = torch.FloatTensor(m)
#		x = torch.add(x,l)
        z = torch.cat((x,y), dim=1)
#		print('x shape after : ', x.shape)
        z = self.fc4(z)
        z = self.relu(z)

        return z

def network(input_cell,hidden_cell,output_cell):

    model = torch.nn.Sequential(
        torch.nn.Linear(input_cell,hidden_cell),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_cell,output_cell)        
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

def network_lin(input_cell,hidden_cell,output_cell):

    model = torch.nn.Sequential(
        torch.nn.Linear(input_cell,hidden_cell),
		#torch.nn.ReLU(),
        torch.nn.Linear(hidden_cell,output_cell),
		#torch.nn.ReLU()
    )
	
    return model

def train(x_train, y_train, network, learning_rate, epochs):
    model = network	
    loss = np.zeros([epochs,1])
    loss_fn = torch.nn.MSELoss()		# Loss function
    for ix in range(epochs):
        y_hat = model(x_train)	# Forward pass
        loss_var = loss_fn(y_hat, y_train) # Compute loss
        loss[ix] = loss_var.data.cpu()
        model.zero_grad()		# Reset gradient
        loss_var.backward()		# Backward pass
        for param in model.parameters():	#Update weights
            param.data = param.data - learning_rate * param.grad.data
#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )	io.write_output(x_train,"x_train")
#	io.write_output(torch.max(y_hat, dim=1)[1],"y_hat")
#	io.write_output(y_train,"y_train")
    return network, loss

def train_all(x_train_1, x_train_2, x_train_3, y_train_1, y_train_2, y_train_3,  network, learning_rate, epochs):
    model = network	
    loss = np.zeros([epochs,1])
    loss_fn = torch.nn.MSELoss()		# Loss function
    for ix in range(epochs):
        y_hat_1 = model(x_train_1)	# Forward pass
        y_hat_2 = model(x_train_2)
        y_hat_3 = model(x_train_3)
        loss_var_1 = loss_fn(y_hat_1, y_train_1) # Compute loss
        loss_var_2 = loss_fn(y_hat_2, y_train_2)
        loss_var_3 = loss_fn(y_hat_3, y_train_3)
        loss_var = loss_var_1 + loss_var_2 + loss_var_3
        loss[ix] = loss_var.data.cpu()
        model.zero_grad()		# Reset gradient
        loss_var.backward()		# Backward pass
        for param in model.parameters():	#Update weights
            param.data = param.data - learning_rate * param.grad.data
#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )	io.write_output(x_train,"x_train")
#	io.write_output(torch.max(y_hat, dim=1)[1],"y_hat")
#	io.write_output(y_train,"y_train")
    return network, loss

def test(x_test, y_test, network):

    Y_hat = network(x_test)
    y_tmp = torch.max(Y_hat, dim=1)[1] #return max index
    y_tmp = y_tmp.data.cpu().numpy()
    acc = np.mean(1 * (y_tmp == y_test))
	
    return acc

def test_b(x_test, y_test, network):

    y_hat = network(x_test)
    y_hat = torch.round(y_hat)
    y_tmp = y_hat.data.cpu().numpy()
    bit = len(y_tmp[0])
    y_hat_1D = np.zeros((len(y_tmp)))
    for l in range(len(y_tmp)):
        for b in range(bit):
            y_hat_1D[l] += (2**b)*y_tmp[l,b]

    acc = np.mean(1 * (y_hat_1D == y_test))
	
    return acc

def quantization_train(x_train, y_train, network, learning_rate, epochs, bit) :

    loss = np.zeros([epochs,1])
    loss_fn = torch.nn.MSELoss()
#	io.write_weight(network, "network_init")
#	network = util.normalize(network)	
#	io.write_weight(network, "network_norm")
    for param in network.parameters():
        param.data = util.quantize(param.data,bit)

#	io.write_weight(network, "network_quant")
    for ix in range(epochs):
        y_hat = network(x_train)	#Forward pass
        loss_var = loss_fn(y_hat, y_train)
        loss[ix] = loss_var.data.cpu()
        network.zero_grad()
        loss_var.backward()
		
        for param in network.parameters():
            param.data = param.data - util.quantize(learning_rate * param.grad.data,bit)

#		network = util.normalize(network)
		
#		for param in network.parameters():
#			param.data = util.quantize(param.data,bit)
	
#		if ix % 1000 == 0 :
#			print('Current epochs : ' + str(ix) + ' th epochs' )

    return network, loss

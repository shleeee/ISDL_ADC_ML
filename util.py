import numpy as np
import torch

def quantize(value,bit):
	n = 2**bit - 1
	return torch.round(value*n)/n

def normalize(network):
	init=0
	for param in network.parameters():
		if(init ==0 ):
			max_value = torch.max(abs(param.data)).item()
			init = 1

		else :
			if(max_value < torch.max(abs(param.data)).item()):
				max_value = torch.max(abs(param.data)).item()

	for param in network.parameters():
		param.data = param.data /max_value

	return network

import numpy as np
import torch

def quantize(value,int_bit,float_bit):
#	if((value < 2**int_bit) and (value > -2**int_bit)):
#		value = value
#	elif(value > (2**int_bit-1)):
#		value = 2**int_bit-1
#	elif(value < (-2**int_bit+1)):
#		value = -2**int_bit+1
	value1 = torch.floor(value)
	value2 = torch.clamp(value1, -2**int_bit+1, 2**int_bit-1)
	value3 = value - value1
	n = 2**float_bit - 1
	value4 = torch.round(value3*n)/n
	return (value2 + value4)

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

	

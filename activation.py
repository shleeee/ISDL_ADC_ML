import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def cReLU(input):

	if input < 4:
		return torch.ReLU(input)
	else :
		return 4

class cReLU(nn.Module):

	def forward(self, input):
		return cReLU(input)

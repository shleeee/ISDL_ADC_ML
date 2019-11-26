import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import data_io as io
import data_io_symbol as ios
import util
import model
from torch.autograd import Variable
import param

Load_PATH = "./model/network.pt"
Save_PATH = "./model/network.pt"

epochs = 5000
epochs_q = 8000
bit = 3
k = 0
lr = 0.5
lr_q = 0.1


X_train=ios.load_input(param.data_out,1,8) # 5x800000
Y_train, y_train = ios.load_output(param.data_in,2) # 2x800000

tic = time.time()

dtype = torch.cuda.FloatTensor

X_train = torch.from_numpy(X_train).type(dtype)
Y_train = torch.from_numpy(Y_train).type(dtype)
finish = 0
iteration = 0
max_value = 0
while(finish == 0 and iteration < 21):
	ML_EQ = model.network(8, 10, 4)

#ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
#ML_EQ.eval()

	ML_EQ = torch.nn.DataParallel(ML_EQ)
	ML_EQ.cuda()

	ML_EQ, loss = model.train(X_train, Y_train, ML_EQ, lr, epochs)
	model.test(X_train, y_train, ML_EQ)

	if(model.test(X_train, y_train, ML_EQ) > max_value):
		max_value = model.test(X_train, y_train, ML_EQ)
	if(model.test(X_train, y_train, ML_EQ)*100 > 99.999):
		finish = 1
	iteration = iteration + 1

print('Training Accuracy: ' + str(100*max_value))
print('iteration : '+str(iteration))
io.write_weight(ML_EQ, "./result/ML_EQ_weight")

#ML_EQ, loss_q = model.quantization_train(X_train, Y_train, ML_EQ, lr_q, epochs_q, bit ) 

#print('Quantization Accuracy: ' + str(model.test(X_train, y_train, ML_EQ)*100))

#io.write_weight(ML_EQ, "./result/ML_EQ_Quantized_weight")

'''
plt.subplot(1,2,1)
ix = np.arange(epochs)
plt.plot(ix,100*loss)
plt.title('Training loss')
plt.ylabel('loss')
'''
print(str(time.time() - tic) + ' s')

#X_test = Variable(torch.from_numpy(X_test).type(dtype), requires_grad = False)
#Y_test = Variable(torch.from_numpy(Y_test).type(dtype), requires_grad = False)

#print('Test Accuracy: ' + str(model.test(X_test, Y_test, ML_EQ)*100))

#torch.save(ML_EQ.module.state_dict(), Save_PATH) # Saving Model
#io.write_weight(ML_EQ, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()

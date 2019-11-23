import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import data_io as io
import util
import model
from torch.autograd import Variable

Load_PATH = "./model/network.pt"
Save_PATH = "./model/network.pt"

epochs = 8000
epochs_q = 10000
bit = 3
lr = 0.5
lr_q = 0.1


#X_train=io.load_input("./data/NRZ/27dB/PRBS31/data_out.txt",6,3) # 30x819126
#Y_train, y_train = io.load_output("./data/NRZ/27dB/PRBS31/data_in.txt",1) # 2x819124

tic = time.time()

dtype = torch.cuda.FloatTensor

#X_train = torch.from_numpy(X_train).type(dtype)
#Y_train = torch.from_numpy(Y_train).type(dtype)

ML_EQ = model.network(18, 10, 2)

ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
ML_EQ.eval()

ML_EQ = torch.nn.DataParallel(ML_EQ)
ML_EQ.cuda()

#ML_EQ, loss = model.train(X_train, Y_train, ML_EQ, lr, epochs)

#print('Training Accuracy: ' + str(model.test(X_train, y_train, ML_EQ)*100))

#ML_EQ, loss_q = model.quantization_train(X_train, Y_train, ML_EQ, lr_q, epochs_q, bit ) 

#print('Quantization Accuracy: ' + str(model.test(X_train, y_train, ML_EQ)*100))

'''
plt.subplot(1,2,1)
ix = np.arange(epochs)
plt.plot(ix,100*loss)
plt.title('Training loss')
plt.ylabel('loss')
'''
print(str(time.time() - tic) + ' s')

for num in range(11) :
 
	X_test=io.load_input("./data/NRZ/23dB/data_out_%d.txt" %num,6,3) # 30x819126
	Y_test, y_test = io.load_output("./data/NRZ/23dB/data_in_%d.txt" %num,1) # 2x819124

	X_test = Variable(torch.from_numpy(X_test).type(dtype), requires_grad = False)
	Y_test = Variable(torch.from_numpy(Y_test).type(dtype), requires_grad = False)

	print(str(num) + ' Test Accuracy :' + str(model.test(X_test, y_test, ML_EQ)*100))

#torch.save(ML_EQ.module.state_dict(), Save_PATH) # Saving Model
#io.write_weight(ML_EQ, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()

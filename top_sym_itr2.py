import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import data_io as io
import util
import model
from torch.autograd import Variable

#Load_PATH = "./model/network.pt"
Save_PATH = "./model/27dB/PRBS31/network_sym.pt"

epochs = 8000
epochs_q = 6000
bit = 3
lr = 0.5
lr_q = 0.1


X_train=io.load_input_sym("./data/PAM4/27dB/data_out_5.txt",6,5) # 30x819126
Y_train, y_train = io.load_output_b("./data/PAM4/27dB/data_in_5.txt",2) # 2x819124

tic = time.time()

dtype = torch.cuda.FloatTensor

X_train = torch.from_numpy(X_train).type(dtype)
Y_train = torch.from_numpy(Y_train).type(dtype)

train_acc = 0
best_model = model.network(5,10,8)



for itr in range(1,21) :

	ML_EQ = model.network(5,16,8)

#ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
#ML_EQ.eval()

	ML_EQ = torch.nn.DataParallel(ML_EQ)
	ML_EQ.cuda()

	ML_EQ, loss = model.train(X_train, Y_train, ML_EQ, lr, epochs)
	
	temp_acc = model.test(X_train, y_train, ML_EQ)*100
	if(temp_acc>train_acc) :
		train_acc = temp_acc
		best_model = ML_EQ
	if(train_acc>99.99999) :
		break


print('train_acc : ' + str(train_acc) + ' %')
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
#print(str(time.time() - tic) + ' s')
print(str(time.time() - tic) + ' s')

torch.save(best_model.state_dict(), Save_PATH) # Saving Model
io.write_weight(best_model, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()


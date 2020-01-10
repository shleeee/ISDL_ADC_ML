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
Save_PATH = "./model/33dB/PRBS31/network.pt"

epochs = 8000
epochs_q = 6000
bit = 3
lr = 0.5
lr_q = 0.1


X_train=io.load_input("./data/NRZ/27dB/PRBS31/data_out_6.txt",6,3) # 30x819126
Y_train, y_train = io.load_output("./data/NRZ/27dB/PRBS31/data_in_6.txt",1) # 2x819124

tic = time.time()

dtype = torch.cuda.FloatTensor

X_train = torch.from_numpy(X_train).type(dtype)
Y_train = torch.from_numpy(Y_train).type(dtype)

train_acc = 0

for itr in range(1,11) :

	ML_EQ = model.network_lin(18, 3, 2)

#ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
#ML_EQ.eval()

	ML_EQ = torch.nn.DataParallel(ML_EQ)
	ML_EQ.cuda()

	ML_EQ, loss = model.train(X_train, Y_train, ML_EQ, lr, epochs)
	
	temp_acc = model.test(X_train, y_train, ML_EQ)*100
	if(temp_acc>train_acc) :
		train_acc = temp_acc
	if(train_acc>99.99999) :
		print('train_acc : ' + str(train_acc) + ' %')
		break

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

for num in range(4,7) :
 	
	X_test=io.load_input("./data/NRZ/27dB/PRBS15_jitter/data_out_%d.txt" %num,6,3) # 30x819126
	Y_test, y_test = io.load_output("./data/NRZ/27dB/PRBS15_jitter/data_in_%d.txt" %num,1) # 2x819124

	X_test = Variable(torch.from_numpy(X_test).type(dtype), requires_grad = False)
	Y_test = Variable(torch.from_numpy(Y_test).type(dtype), requires_grad = False)
	
	num_itr = 0
	test_acc = 0

	#ML_EQ = model.network(18 , 3, 2)
	#ML_EQ.load_state_dict(torch.load("./model/23dB/PRBS31/network_%d.pt" %num))
	#ML_EQ.eval()

	#ML_EQ = torch.nn.DataParallel(ML_EQ)
	#ML_EQ.cuda()	
	print(str(num)+'-th UI input')

	for itr in range(1,2) :	
		num_itr += 1
		temp_acc = model.test(X_test, y_test, ML_EQ)*100
#		print(str(itr) + ' th iteration for ' + str(num) + '-th input')
		if(temp_acc>test_acc) :
			test_acc = temp_acc
		if(test_acc>99.99999) :
			break

#	print('iteration : ' + str(num_itr))		
	print(str(time.time() - tic) + ' s')
	print('Test Acurracy : ' + str(test_acc) + ' %' )

print(str(time.time() - tic) + ' s')
'''
torch.save(ML_EQ.module.state_dict(), Save_PATH) # Saving Model
#io.write_weight(ML_EQ, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()
'''

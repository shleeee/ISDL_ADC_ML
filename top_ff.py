import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
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

epochs = 16000
epochs_q = 8000
int_bit = 4
float_bit = 2
k = 0
lr = 0.5
lr_q = 0.1

X_train=ios.load_input(param.data_out1,1,5) # 5x800000
X_train2=ios.load_input(param.data_out2,1,5) # 5x800000
#X_train3=ios.load_input(param.data_out3,1,5) # 5x800000
Y_train, y_train = ios.load_output(param.data_in1,2) # 2x800000
Y_train2, y_train2 = ios.load_output(param.data_in2,2) # 2x800000
#Y_train3, y_train3 = ios.load_output(param.data_in3,2) # 2x800000
#X_train=ios.load_input("./data/PAM4/all9/data_out_27dB.txt",1,5) # 5x800000
#X_train2=ios.load_input("./data/PAM4/all9/data_out_23dB.txt",1,5) # 5x800000
#X_train3=ios.load_input("./data/PAM4/all9/data_out_17dB.txt",1,5) # 5x800000
#Y_train, y_train = ios.load_output("./data/PAM4/all9/data_in_27dB.txt",2) # 2x800000
#Y_train2, y_train2 = ios.load_output("./data/PAM4/all9/data_in_23dB.txt",2) # 2x800000
#Y_train3, y_train3 = ios.load_output("./data/PAM4/all9/data_in_17dB.txt",2) # 2x800000

#X_test=ios.load_input(param.data_out_test,1,5)
#Y_test, y_test = ios.load_output(param.data_in_test,2)
#X_test=ios.load_input("./data/PAM4/all9/test/data_out_27dB.txt",1,5)
#Y_test, y_test = ios.load_output("./data/PAM4/all9/test/data_in_27dB.txt",2)
#X_test2=ios.load_input("./data/PAM4/all9/test/data_out_23dB.txt",1,5)
#Y_test2, y_test2 = ios.load_output("./data/PAM4/all9/test/data_in_23dB.txt",2)
#X_test3=ios.load_input("./data/PAM4/all9/test/data_out_17dB.txt",1,5)
#Y_test3, y_test3 = ios.load_output("./data/PAM4/all9/test/data_in_17dB.txt",2)

tic = time.time()

dtype = torch.cuda.FloatTensor

X_train = torch.from_numpy(X_train).type(dtype)
Y_train = torch.from_numpy(Y_train).type(dtype)
X_train2 = torch.from_numpy(X_train2).type(dtype)
Y_train2 = torch.from_numpy(Y_train2).type(dtype)
#X_train3 = torch.from_numpy(X_train3).type(dtype)
#Y_train3 = torch.from_numpy(Y_train3).type(dtype)
#X_test = torch.from_numpy(X_test).type(dtype)
#Y_test = torch.from_numpy(Y_test).type(dtype)
#X_test = Variable(torch.from_numpy(X_test).type(dtype), requires_grad = False)
#Y_test = Variable(torch.from_numpy(Y_test).type(dtype), requires_grad = False)
#X_test2 = Variable(torch.from_numpy(X_test2).type(dtype), requires_grad = False)
#Y_test2 = Variable(torch.from_numpy(Y_test2).type(dtype), requires_grad = False)
#X_test3 = Variable(torch.from_numpy(X_test3).type(dtype), requires_grad = False)
#Y_test3 = Variable(torch.from_numpy(Y_test3).type(dtype), requires_grad = False)
#X_test3 = Variable(torch.from_numpy(X_test3).type(dtype), requires_grad = False)
#Y_test3 = Variable(torch.from_numpy(Y_test3).type(dtype), requires_grad = False)
#X_train2 = torch.from_numpy(X_train2).type(dtype)
#Y_train2 = torch.from_numpy(Y_train2).type(dtype)
finish = 0
iteration = 0
max_value = 0
while(finish == 0 and iteration < 21):
#	model.k = 0
#	ML_EQ1 = model.network(8, 10, 4)
	ML_EQ = model.FFClassifier(5, 5, 5, 4)
#	print(ML_EQ1)
#	print(ML_EQ)

#ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
#ML_EQ.eval()

	ML_EQ = torch.nn.DataParallel(ML_EQ)
	ML_EQ.cuda()
#	ML_EQ, loss = model.train(X_train, Y_train, ML_EQ, lr, epochs)

	ML_EQ, loss = model.train2(X_train, Y_train, X_train2, Y_train2, ML_EQ, lr, epochs)
	#model.k = 1
	#ML_EQ, loss_q = model.quantization_train3(X_train, Y_train, X_train2, Y_train2, X_train3, Y_train3, ML_EQ, lr_q, epochs_q, int_bit, float_bit) 

	if((model.test(X_train, y_train, ML_EQ)+model.test(X_train2, y_train2, ML_EQ))/3.0 > max_value):
		max_value = (model.test(X_train, y_train, ML_EQ)+model.test(X_train2, y_train2, ML_EQ))/3.0
		ML_EQ_best = ML_EQ
	if((model.test(X_train, y_train, ML_EQ)+model.test(X_train2, y_train2, ML_EQ))/3.0*100 > 99.9999):
		finish = 1
	iteration = iteration + 1

print('Training Accuracy: ' + str(100*max_value))
print('iteration : '+str(iteration))
io.write_weight(ML_EQ_best, "./result/ML_EQ_weight")

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
#print('Test Accuracy: ' + str(model.test(X_test, y_test, ML_EQ_best)*100))
print('23dB training Accuracy: ' + str(model.test(X_train, y_train, ML_EQ_best)*100))
print('17dB training Accuracy: ' + str(model.test(X_train2, y_train2, ML_EQ_best)*100))
#print('10dB training Accuracy: ' + str(model.test(X_train3, y_train3, ML_EQ_best)*100))
#print('27dB test Accuracy: ' + str(model.test(X_test, y_test, ML_EQ_best)*100))
#print('23dB test Accuracy: ' + str(model.test(X_test2, y_test2, ML_EQ_best)*100))
#print('17dB test Accuracy: ' + str(model.test(X_test3, y_test3, ML_EQ_best)*100))

print(str(time.time() - tic) + ' s')

#torch.save(ML_EQ.module.state_dict(), Save_PATH) # Saving Model
#io.write_weight(ML_EQ, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

epochs = 15000
epochs_q = 6000
bit = 3
lr = 0.5
lr_q = 0.1


X_train_1=io.load_input_sym("./data/PAM4/27dB/PRBS710/train/data_out_5.txt",6,5) # 30x819126
X_train_2=io.load_input_sym("./data/PAM4/23dB/train/data_out_5.txt",6,5) # 30x819126
X_train_3=io.load_input_sym("./data/PAM4/17dB/train/data_out_5.txt",6,5) # 30x819126


Y_train_1, y_train_1 = io.load_output("./data/PAM4/27dB/PRBS710/train/data_in_5.txt",2) # 2x819124
Y_train_2, y_train_2 = io.load_output("./data/PAM4/23dB/train/data_in_5.txt",2) # 2x819124
Y_train_3, y_train_3 = io.load_output("./data/PAM4/17dB/train/data_in_5.txt",2) # 2x819124



tic = time.time()

dtype = torch.cuda.FloatTensor

X_train_1 = torch.from_numpy(X_train_1).type(dtype)
X_train_2 = torch.from_numpy(X_train_2).type(dtype)
X_train_3 = torch.from_numpy(X_train_3).type(dtype)
Y_train_1 = torch.from_numpy(Y_train_1).type(dtype)
Y_train_2 = torch.from_numpy(Y_train_2).type(dtype)
Y_train_3 = torch.from_numpy(Y_train_3).type(dtype)

train_acc_1 = 0
train_acc_2 = 0
train_acc_3 = 0
train_acc = 0
best_model = model.network(5,10,4)



for itr in range(1,21) :

    ML_EQ = model.network(5,10,4)

#ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
#ML_EQ.eval()

    ML_EQ = torch.nn.DataParallel(ML_EQ)
    ML_EQ.cuda()

    ML_EQ, loss = model.train_all(X_train_1, X_train_2, X_train_3, Y_train_1, Y_train_2, Y_train_3, ML_EQ, lr, epochs)
	
    temp_acc_1 = model.test(X_train_1, y_train_1, ML_EQ)*100
    temp_acc_2 = model.test(X_train_2, y_train_2, ML_EQ)*100
    temp_acc_3 = model.test(X_train_3, y_train_3, ML_EQ)*100
    temp_acc = temp_acc_1/3 + temp_acc_2/3 + temp_acc_3/3
    if(temp_acc>train_acc) :
        train_acc = temp_acc
        train_acc_1 = temp_acc_1
        train_acc_2 = temp_acc_2
        train_acc_3 = temp_acc_3
        best_model = ML_EQ
    if(train_acc>99.99) :
        break


print('train_acc_1 : ' + str(train_acc_1) + ' %')
print('train_acc_2 : ' + str(train_acc_2) + ' %')
print('train_acc_3 : ' + str(train_acc_3) + ' %')
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

#torch.save(best_model.state_dict(), Save_PATH) # Saving Model
#io.write_weight(best_model, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()


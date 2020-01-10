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


#RNN
epochs = 2000
lr = 0.5

X_train=io.load_input("./data/PAM4/27dB/PRBS710/train/data_out_5.txt",6,1) # 30x819126
Y_train, y_train = io.load_output("./data/PAM4/27dB/PRBS710/train/data_in_5.txt",2) # 2x819124
X_rnn = np.zeros((1,len(X_train),len(X_train[0])))
#Y_rnn = np.zeros((1,len(Y_train),len(Y_train[0])))



X_rnn[0] = X_train

tic = time.time()

dtype = torch.cuda.LongTensor

X_rnn = torch.from_numpy(X_rnn).type(dtype)
Y_train = torch.from_numpy(Y_train).type(dtype)

ML_EQ_rnn = torch.nn.RNN(input_size = 6, hidden_size = 4, num_layers = 1, nonlinearity = 'relu', batch_first = True)
ML_EQ_rnn = torch.nn.DataParallel(ML_EQ_rnn)
ML_EQ_rnn.cuda()

# Training

loss_fn = torch.nn.CrossEntropyLoss()

for ix in range(epochs):
    Y_temp, hidden =  ML_EQ_rnn(X_rnn)
    Y_hat=Y_temp[0,4:,:]
    print(Y_hat)
    print(Y_train)
    loss_var = loss_fn(Y_hat, Y_train)
    ML_EQ_rnn.zero_grad()
    loss_var.backward()

    for param in ML_EQ_rnn.parameters():
        param.data = param.data - lr*param.grad.data
# Test

Y_temp, hidden = ML_EQ_rnn(X_rnn)
Y_hat = Y_temp[4:]
y_hat = torch.max(Y_hat,dim=1)[1]
y_hat = y_hat.data.cpu().numpy()
acc = np.mean(1 * (y_hat == y_train))

print(acc*100)
 

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
import model_ff
from torch.autograd import Variable
import param

Load_PATH = "./model_ff/network.pt"
Save_PATH = "./model_ff/network.pt"

epochs = 5000
epochs_q = 8000
int_bit = 4
float_bit = 2
k = 0
lr = 0.5
lr_q = 0.1

#X_train=ios.load_input(param.data_out1,1,5) # 5x800000
#X_train2=ios.load_input(param.data_out2,1,5) # 5x800000
#X_train3=ios.load_input(param.data_out3,1,5) # 5x800000
#X_train4=ios.load_input(param.data_out4,1,5) # 5x800000
#Y_train, y_train = ios.load_output(param.data_in1,2) # 2x800000
#Y_train2, y_train2 = ios.load_output(param.data_in2,2) # 2x800000
#Y_train3, y_train3 = ios.load_output(param.data_in3,2) # 2x800000
#Y_train4, y_train4 = ios.load_output(param.data_in4,2) # 2x800000
X_train=ios.load_input("./data/PAM4_real/w_AGC/17dB/data_out_7.txt",1,5) # 5x800000
#X_train2=ios.load_input("./data/PAM4/all9/data_out_23dB.txt",1,5) # 5x800000
#X_train3=ios.load_input("./data/PAM4/all9/data_out_17dB.txt",1,5) # 5x800000
Y_train, y_train = ios.load_output("./data/PAM4_real/w_AGC/17dB/data_in_7.txt",2) # 2x800000
#Y_train2, y_train2 = ios.load_output("./data/PAM4/all9/data_in_23dB.txt",2) # 2x800000
#Y_train3, y_train3 = ios.load_output("./data/PAM4/all9/data_in_17dB.txt",2) # 2x800000

X_test1=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_1dB.txt",1,5) # 5x800000
X_test2=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_2dB.txt",1,5) # 5x800000
X_test3=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_3dB.txt",1,5) # 5x800000
X_test4=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_4dB.txt",1,5) # 5x800000
X_test5=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_5dB.txt",1,5) # 5x800000
X_test6=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_6dB.txt",1,5) # 5x800000
X_test7=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_7dB.txt",1,5) # 5x800000
X_test8=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_8dB.txt",1,5) # 5x800000
X_test9=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_9dB.txt",1,5) # 5x800000
X_test10=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_10dB.txt",1,5) # 5x800000
X_test11=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_11dB.txt",1,5) # 5x800000
X_test12=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_12dB.txt",1,5) # 5x800000
X_test13=ios.load_input("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_out_13dB.txt",1,5) # 5x800000
Y_test1, y_test1 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_1dB.txt",2) # 2x800000
Y_test2, y_test2 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_2dB.txt",2) # 2x800000
Y_test3, y_test3 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_3dB.txt",2) # 2x800000
Y_test4, y_test4 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_4dB.txt",2) # 2x800000
Y_test5, y_test5 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_5dB.txt",2) # 2x800000
Y_test6, y_test6 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_6dB.txt",2) # 2x800000
Y_test7, y_test7 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_7dB.txt",2) # 2x800000
Y_test8, y_test8 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_8dB.txt",2) # 2x800000
Y_test9, y_test9 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_9dB.txt",2) # 2x800000
Y_test10, y_test10 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_10dB.txt",2) # 2x800000
Y_test11, y_test11 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_11dB.txt",2) # 2x800000
Y_test12, y_test12 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_12dB.txt",2) # 2x800000
Y_test13, y_test13 = ios.load_output("./data/PAM4_real/w_AGC/17dB/SNR/prbs7/data_in_13dB.txt",2) # 2x800000
#Y_train_ch, y_train_ch = ios.load_output("./data/PAM4/all3/ch_27dB.txt",2) # 2x800000
#Y_train2_ch, y_train2_ch = ios.load_output("./data/PAM4/all3/ch_23dB.txt",2) # 2x800000
#Y_train3_ch, y_train3_ch = ios.load_output("./data/PAM4/all3/ch_17dB.txt",2) # 2x800000
#Y_train4_ch, y_train4_ch = ios.load_output("./data/PAM4/all3/ch_10dB.txt",2) # 2x800000
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
#X_train2 = torch.from_numpy(X_train2).type(dtype)
#Y_train2 = torch.from_numpy(Y_train2).type(dtype)
#X_train3 = torch.from_numpy(X_train3).type(dtype)
#Y_train3 = torch.from_numpy(Y_train3).type(dtype)
#X_train4 = torch.from_numpy(X_train4).type(dtype)
#Y_train4 = torch.from_numpy(Y_train4).type(dtype)
#Y_train_ch = torch.from_numpy(Y_train_ch).type(dtype)
#Y_train2_ch = torch.from_numpy(Y_train2_ch).type(dtype)
#Y_train3_ch = torch.from_numpy(Y_train3_ch).type(dtype)
#Y_train4_ch = torch.from_numpy(Y_train4_ch).type(dtype)
#X_test = torch.from_numpy(X_test).type(dtype)
#Y_test = torch.from_numpy(Y_test).type(dtype)
#X_test = Variable(torch.from_numpy(X_test).type(dtype), requires_grad = False)
#Y_test = Variable(torch.from_numpy(Y_test).type(dtype), requires_grad = False)
X_test1 = Variable(torch.from_numpy(X_test1).type(dtype), requires_grad = False)
X_test2 = Variable(torch.from_numpy(X_test2).type(dtype), requires_grad = False)
X_test3 = Variable(torch.from_numpy(X_test3).type(dtype), requires_grad = False)
X_test4 = Variable(torch.from_numpy(X_test4).type(dtype), requires_grad = False)
X_test5 = Variable(torch.from_numpy(X_test5).type(dtype), requires_grad = False)
X_test6 = Variable(torch.from_numpy(X_test6).type(dtype), requires_grad = False)
X_test7 = Variable(torch.from_numpy(X_test7).type(dtype), requires_grad = False)
X_test8 = Variable(torch.from_numpy(X_test8).type(dtype), requires_grad = False)
X_test9 = Variable(torch.from_numpy(X_test9).type(dtype), requires_grad = False)
X_test10 = Variable(torch.from_numpy(X_test10).type(dtype), requires_grad = False)
X_test11 = Variable(torch.from_numpy(X_test11).type(dtype), requires_grad = False)
X_test12 = Variable(torch.from_numpy(X_test12).type(dtype), requires_grad = False)
X_test13 = Variable(torch.from_numpy(X_test13).type(dtype), requires_grad = False)
Y_test1 = Variable(torch.from_numpy(Y_test1).type(dtype), requires_grad = False)
Y_test2 = Variable(torch.from_numpy(Y_test2).type(dtype), requires_grad = False)
Y_test3 = Variable(torch.from_numpy(Y_test3).type(dtype), requires_grad = False)
Y_test4 = Variable(torch.from_numpy(Y_test4).type(dtype), requires_grad = False)
Y_test5 = Variable(torch.from_numpy(Y_test5).type(dtype), requires_grad = False)
Y_test6 = Variable(torch.from_numpy(Y_test6).type(dtype), requires_grad = False)
Y_test7 = Variable(torch.from_numpy(Y_test7).type(dtype), requires_grad = False)
Y_test8 = Variable(torch.from_numpy(Y_test8).type(dtype), requires_grad = False)
Y_test9 = Variable(torch.from_numpy(Y_test9).type(dtype), requires_grad = False)
Y_test10 = Variable(torch.from_numpy(Y_test10).type(dtype), requires_grad = False)
Y_test11 = Variable(torch.from_numpy(Y_test11).type(dtype), requires_grad = False)
Y_test12 = Variable(torch.from_numpy(Y_test12).type(dtype), requires_grad = False)
Y_test13 = Variable(torch.from_numpy(Y_test13).type(dtype), requires_grad = False)
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
#max_ch = 0

#while(finish == 0 and iteration < 31):
##	model_ff.k = 0
#	ML_CH = model_ff.network(5, 10, 4)
#	ML_CH = torch.nn.DataParallel(ML_CH)
#	ML_CH.cuda()
#	ML_CH, loss = model_ff.train4(X_train, Y_train_ch, X_train2, Y_train2_ch, X_train3, Y_train3_ch, X_train4, Y_train4_ch, ML_CH, lr, epochs)
#	if((model_ff.test(X_train, y_train_ch, ML_CH)+model_ff.test(X_train2, y_train2_ch, ML_CH)+model_ff.test(X_train3, y_train3_ch, ML_CH)+model_ff.test(X_train4, y_train4_ch, ML_CH))/4.0 > max_ch):
#		max_ch = (model_ff.test(X_train, y_train_ch, ML_CH)+model_ff.test(X_train2, y_train2_ch, ML_CH)+model_ff.test(X_train3, y_train3_ch, ML_CH)+model_ff.test(X_train4, y_train4_ch, ML_CH))/4.0
#		ML_CH_best = ML_CH
#	if((model_ff.test(X_train, y_train_ch, ML_CH)+model_ff.test(X_train2, y_train2_ch, ML_CH)+model_ff.test(X_train3, y_train3_ch, ML_CH)+model_ff.test(X_train4, y_train4_ch, ML_CH))/4.0*100 > 95.9999):
#		finish = 1
#	iteration = iteration + 1
#
#iteration = 0
#finish = 0
#
#X_train_mid = ML_CH_best(X_train)
#X_train2_mid = ML_CH_best(X_train2)
#X_train3_mid = ML_CH_best(X_train3)
#X_train4_mid = ML_CH_best(X_train4)

#print('Channel selection training!')
#print('23dB : ', model_ff.test(X_train, y_train_ch, ML_CH_best))
#print('17dB : ', model_ff.test(X_train2, y_train2_ch, ML_CH_best))
#print('10dB : ', model_ff.test(X_train3, y_train3_ch, ML_CH_best))
#print('6dB : ', model_ff.test(X_train4, y_train4_ch, ML_CH_best))
#
#print('X_train_mid : ', X_train_mid)
#print('X_train2_mid : ', X_train2_mid)
#print('X_train3_mid : ', X_train3_mid)
#print('X_train4_mid : ', X_train4_mid)
#
#X_train = torch.cat((X_train,X_train_mid), dim=1)
#X_train2 = torch.cat((X_train2,X_train2_mid), dim=1)
#X_train3 = torch.cat((X_train3,X_train3_mid), dim=1)
#X_train4 = torch.cat((X_train4,X_train4_mid), dim=1)

#X_train_mid = torch.from_numpy(X_train_mid).type(dtype)
#X_train2_mid = torch.from_numpy(X_train2_mid).type(dtype)
#X_train3_mid = torch.from_numpy(X_train3_mid).type(dtype)
#X_train4_mid = torch.from_numpy(X_train4_mid).type(dtype)

while(finish == 0 and iteration < 31):
#	model_ff.k = 0
	ML_EQ = model_ff.network(5, 3, 4)
	ML_EQ = torch.nn.DataParallel(ML_EQ)
	ML_EQ.cuda()
	ML_EQ, loss = model_ff.train(X_train, Y_train, ML_EQ, lr, epochs)
	ML_EQ, loss_q = model_ff.quantization_train(X_train, Y_train, ML_EQ, lr_q, epochs_q, int_bit, float_bit) 
	if(model_ff.test(X_train, y_train, ML_EQ) > max_value):
		max_value = model_ff.test(X_train, y_train, ML_EQ)
		ML_EQ_best = ML_EQ
	if(100*model_ff.test(X_train, y_train, ML_EQ) > 99.9999):
		finish = 1
	iteration = iteration + 1


#while(finish == 0 and iteration < 31):
##	model_ff.k = 0
##	ML_EQ = model_ff.network(5, 10, 4)
#	ML_EQ = model_ff.FFClassifier(5, 5, 4)
##	print(ML_EQ1)
##	print(ML_EQ)
#
##ML_EQ.load_state_dict(torch.load(Load_PATH)) # Loading Model
##ML_EQ.eval()
#
#	ML_EQ = torch.nn.DataParallel(ML_EQ)
#	ML_EQ.cuda()
##	ML_EQ, loss = model_ff.train(X_train, Y_train, ML_EQ, lr, epochs)
#
#	ML_EQ, loss = model_ff.train4_up(X_train, X_train_mid, Y_train, X_train2, X_train2_mid, Y_train2, X_train3, X_train3_mid, Y_train3, X_train4, X_train4_mid, Y_train4, ML_EQ, lr, epochs)
#	#model_ff.k = 1
#	#ML_EQ, loss_q = model_ff.quantization_train3(X_train, Y_train, X_train2, Y_train2, X_train3, Y_train3, ML_EQ, lr_q, epochs_q, int_bit, float_bit) 
#	if((model_ff.test_up(X_train, X_train_mid, y_train_ch, ML_EQ)+model_ff.test_up(X_train2, X_train2_mid, y_train2_ch, ML_EQ)+model_ff.test_up(X_train3, X_train3_mid, y_train3_ch, ML_EQ)+model_ff.test_up(X_train4, X_train4_mid, y_train4_ch, ML_EQ))/4.0 > max_value):
#		max_value = (model_ff.test_up(X_train, X_train_mid, y_train_ch, ML_EQ)+model_ff.test_up(X_train2, X_train2_mid, y_train2_ch, ML_EQ)+model_ff.test_up(X_train3, X_train3_mid, y_train3_ch, ML_EQ)+model_ff.test_up(X_train4, X_train4_mid, y_train4_ch, ML_EQ))/4.0
#		ML_EQ_best = ML_EQ
#	if((model_ff.test_up(X_train, X_train_mid, y_train_ch, ML_EQ)+model_ff.test_up(X_train2, X_train2_mid, y_train2_ch, ML_EQ)+model_ff.test_up(X_train3, X_train3_mid, y_train3_ch, ML_EQ)+model_ff.test_up(X_train4, X_train4_mid, y_train4_ch, ML_EQ))/4.0*100 > 95.9999):
#		finish = 1
#	iteration = iteration + 1

print('Training Accuracy: ' + str(100*max_value))
print('iteration : '+str(iteration))
io.write_weight(ML_EQ_best, "./result/ML_EQ_weight")

#ML_EQ, loss_q = model_ff.quantization_train(X_train, Y_train, ML_EQ, lr_q, epochs_q, bit ) 

#print('Quantization Accuracy: ' + str(model_ff.test(X_train, y_train, ML_EQ)*100))

#io.write_weight(ML_EQ, "./result/ML_EQ_Quantized_weight")

'''
plt.subplot(1,2,1)
ix = np.arange(epochs)
plt.plot(ix,100*loss)
plt.title('Training loss')
plt.ylabel('loss')
'''
#print('Test Accuracy: ' + str(model_ff.test(X_test, y_test, ML_EQ_best)*100))
#print('23dB training Accuracy: ' + str(model_ff.test(X_train, y_train, ML_EQ_best)*100))
#print('10dB training Accuracy: ' + str(model_ff.test(X_train3, y_train3, ML_EQ_best)*100))
#print('6dB training Accuracy: ' + str(model_ff.test(X_train4, y_train4, ML_EQ_best)*100))
#print('27dB test Accuracy: ' + str(model_ff.test(X_test, y_test, ML_EQ_best)*100))
#print('23dB test Accuracy: ' + str(model_ff.test(X_test2, y_test2, ML_EQ_best)*100))
#print('17dB test Accuracy: ' + str(model_ff.test(X_test3, y_test3, ML_EQ_best)*100))
print('SNR 1dB test Accuracy: ' + str(model_ff.test(X_test1, y_test1, ML_EQ_best)*100))
print('SNR 2dB test Accuracy: ' + str(model_ff.test(X_test2, y_test2, ML_EQ_best)*100))
print('SNR 3dB test Accuracy: ' + str(model_ff.test(X_test3, y_test3, ML_EQ_best)*100))
print('SNR 4dB test Accuracy: ' + str(model_ff.test(X_test4, y_test4, ML_EQ_best)*100))
print('SNR 5dB test Accuracy: ' + str(model_ff.test(X_test5, y_test5, ML_EQ_best)*100))
print('SNR 6dB test Accuracy: ' + str(model_ff.test(X_test6, y_test6, ML_EQ_best)*100))
print('SNR 7dB test Accuracy: ' + str(model_ff.test(X_test7, y_test7, ML_EQ_best)*100))
print('SNR 8dB test Accuracy: ' + str(model_ff.test(X_test8, y_test8, ML_EQ_best)*100))
print('SNR 9dB test Accuracy: ' + str(model_ff.test(X_test9, y_test9, ML_EQ_best)*100))
print('SNR 10dB test Accuracy: ' + str(model_ff.test(X_test10, y_test10, ML_EQ_best)*100))
print('SNR 11dB test Accuracy: ' + str(model_ff.test(X_test11, y_test11, ML_EQ_best)*100))
print('SNR 12dB test Accuracy: ' + str(model_ff.test(X_test12, y_test12, ML_EQ_best)*100))
print('SNR 13dB test Accuracy: ' + str(model_ff.test(X_test13, y_test13, ML_EQ_best)*100))

print(str(time.time() - tic) + ' s')

#torch.save(ML_EQ.module.state_dict(), Save_PATH) # Saving Model
#io.write_weight(ML_EQ, "./result/ML_EQ_weight")
#io.write_output(ML_EQ, "./result/ML_EQ_output")

#plt.show()

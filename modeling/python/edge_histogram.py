#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

fdah = file('./data_before_hist.txt', 'r')
fdas = file( './data_before_std.txt', 'r')
fdbh = file( './data_after_lineq_hist.txt', 'r')
fdbs = file(  './data_after_lineq_std.txt', 'r')
fdch = file( './data_after_dfe_hist.txt', 'r')
fdcs = file(  './data_after_dfe_std.txt', 'r')
#febh = file('./edge_before_hist.txt', 'r')
#febs = file( './edge_before_std.txt', 'r')
#feah = file( './edge_after_hist.txt', 'r')
#feas = file(  './edge_after_std.txt', 'r')
data_as = []
for line in fdah :
	data_as.append(float(line))

data_as_array=np.array(data_as)
laba='data std before lineq : %1.4f'%data_as_array.std()
print 'data std before lineq : %1.4f' %data_as_array.std()

data_bs = []
for line in fdbh :
	data_bs.append(float(line))

data_bs_array=np.array(data_bs)
labb='data std after lineq before DFE: %1.4f'%data_bs_array.std()
print 'data std after lineq before DFE : %1.4f' %data_bs_array.std()

data_cs = []
for line in fdch :
	data_cs.append(float(line))

data_cs_array=np.array(data_cs)
labc='data std after DFE: %1.4f'%data_cs_array.std()
print 'data std after DFE : %1.4f' %data_cs_array.std()

data_ah = []
data_bh = []
data_ch = []
for line in fdas :
	data_ah.append(float(line))

for line in fdbs :
	data_bh.append(float(line))

for line in fdcs :
	data_ch.append(float(line))

plt.figure(1)
plt.title('data histogram result before equalizing')
h, b ,p = plt.hist(data_ah, 100, range=[-0.5,0.5], normed=1, color='blue', label=laba)
plt.xlabel('voltage[V]')
plt.ylabel('normalized bins')
plt.legend(loc=2)
plt.savefig('data_hist_before_lineq.png')

plt.figure(2)
plt.title('data histogram result after lineq')
h, b ,p = plt.hist(data_bh, 100, range=[-0.5,0.5], normed=1, color='red', label=labb)
plt.xlabel('voltage[V]')
plt.ylabel('normalized bins')
plt.legend(loc=2)
plt.savefig('data_hist_after_lineq.png')

plt.figure(3)
plt.title('data histogram result after DFE')
h, b ,p = plt.hist(data_ch, 100, range=[-0.5,0.5], normed=1, color='green', label=labc)
plt.xlabel('voltage[V]')
plt.ylabel('normalized bins')
plt.legend(loc=2)
plt.savefig('data_hist_after_DFE.png')

plt.figure(4)
plt.title('data histogram compare result ')
h, b ,p = plt.hist(data_ah, 100, range=[-0.5,0.5], normed=1, color='blue' , alpha=0.5,  label=laba)
h, b ,p = plt.hist(data_bh, 100, range=[-0.5,0.5], normed=1, color='red'  , alpha=0.5,  label=labb)
h, b ,p = plt.hist(data_ch, 100, range=[-0.5,0.5], normed=1, color='green', alpha=0.5,  label=labc)
plt.xlabel('voltage[V]')
plt.ylabel('normalized bins')
plt.legend(loc=2)
plt.savefig('data_hist_compare.png')

#edge = []
#for line in febh :
#	edge.append(float(line))
#
#plt.figure(str(3))
#plt.title('edge result before equalizing')
#h, b ,p = plt.hist(edge, 100, range=[0.3,0.8], normed=1)
#plt.savefig('edge_hist_before_eq.png')
#
#edge = []
#for line in febs :
#	edge.append(float(line))
#
#edge_array=np.array(edge)
#print 'edge std before equalizing : %1.4f' %edge_array.std()
#
#
#edge = []
#for line in feah :
#	edge.append(float(line))
#
#plt.figure(str(4))
#plt.title('edge result after equalizing')
#h, b ,p = plt.hist(edge, 100, range=[0.3,0.8], normed=1)
#plt.savefig('edge_hist_after_eq.png')
#
#edge = []
#for line in feas :
#	edge.append(float(line))
#
#edge_array=np.array(edge)
#print 'edge std after equalizing : %1.4f' %edge_array.std()

plt.show()

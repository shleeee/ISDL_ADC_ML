#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

fjitta = file( './jitt_before.txt', 'r')
fjittb = file(  './jitt_lineq.txt', 'r')
fjittc = file(    './jitt_dfe.txt', 'r')

div=24
data_a = []
i=0
for line in fjitta :
	if(float(line)<0.05 and float(line)>-0.05) :
		data_a.append(float(line)*1000)
	i=i+1;
data_a_array = np.array(data_a)
plt.figure(1)
laba='jitter before EQ = %1.4ffs'%(data_a_array.std())
h, b ,p = plt.hist(data_a, div, range=[-50,50], normed=1, color='blue', label=laba)
plt.xlabel('time,fs', fontsize='xx-large')
plt.ylabel('normalized bins', fontsize='xx-large')
leg=plt.legend(loc=2)
leg.get_frame().set_alpha(0.5)
plt.savefig('jitt_before.png')
print 'before std : %1.4f' % data_a_array.std()

data_b = []
i=0
for line in fjittb :
	if(float(line)<0.05 and float(line)>-0.05) :
		data_b.append(float(line)*1000)
	i=i+1;
data_b_array = np.array(data_b)
plt.figure(2)
labb='jitter after linEQ = %1.4ffs'%(data_b_array.std())
h, b ,p = plt.hist(data_b, div, range=[-50,50], normed=1, color='red', label=labb)
plt.xlabel('time,fs', fontsize='xx-large')
plt.ylabel('normalized bins', fontsize='xx-large')
leg=plt.legend(loc=2)
leg.get_frame().set_alpha(0.5)
plt.savefig('jitt_after_lineq.png')
print 'after linEQ std : %1.4f' % data_b_array.std()

data_c = []
i=0
for line in fjittc :
	if(float(line)<0.05 and float(line)>-0.05) :
		data_c.append(float(line)*1000)
	i=i+1;
data_c_array = np.array(data_c)
plt.figure(3)
labc='jitter after DFE = %1.4ffs'%(data_c_array.std())
h, b ,p = plt.hist(data_c, div, range=[-50,50], normed=1, color='green', label=labc)
plt.xlabel('time,fs', fontsize='xx-large')
plt.ylabel('normalized bins', fontsize='xx-large')
leg=plt.legend(loc=2)
leg.get_frame().set_alpha(0.5)
plt.savefig('jitt_after_dfe.png')
print 'after DFE std : %1.4f' % data_c_array.std()

plt.figure(4)
h, b ,p = plt.hist(data_a, div, range=[-50,50], normed=1, color='blue' , alpha=0.5, label=laba)
h, b ,p = plt.hist(data_b, div, range=[-50,50], normed=1, color='red'  , alpha=0.5, label=labb)
h, b ,p = plt.hist(data_c, div, range=[-50,50], normed=1, color='green', alpha=0.5, label=labc)
plt.xlabel('time,fs', fontsize='xx-large')
plt.ylabel('normalized bins', fontsize='xx-large')
leg=plt.legend(loc=2)
leg.get_frame().set_alpha(0.5)
plt.savefig('jitt_comp.png')

plt.show()

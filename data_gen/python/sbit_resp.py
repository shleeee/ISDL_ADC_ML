#!/usr/bin/env python

f1bit = file(  './1bit_resp.txt', 'r')

ratio_volt = 1000
ratio_time = 1
ready = 0
for line in feye :
	if(ready!=0) :
		time.append(((i-bias)%period)/ratio_time)
		vol.append(float(line)*ratio_volt)
	if(i==(bias+period-1)) :
		i=bias
		ready=ready+1
	else :
		i=i+1

time_array = np.array(time)
vol_array = np.array(vol)

plt.figure(1)
plt.axis([0,200,-500,500])
plt.hexbin(time_array, vol_array, bins='log', extent=[0,200,-500,500])
plt.title('transmitting EYE diagram with log scale')
plt.xlabel('time[ps]')
plt.ylabel('p2p voltage[mV]')
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.savefig('eye_diagram_tx.png')


#vol  = []
#time = []
#feye = file(  './file_for_eye_pre_eq.txt', 'r')
#ready = 0
#i=0
#
#ratio_volt = 1000
#ratio_time = 1
#ready = 0
#for line in feye :
#	if(ready!=0) :
#		time.append(((i-bias)%period)/ratio_time)
#		vol.append(float(line)*ratio_volt)
#	if(i==(bias+period-1)) :
#		i=bias
#		ready=ready+1
#	else :
#		i=i+1
#
#time_array = np.array(time)
#vol_array = np.array(vol)
#
#plt.figure(2)
#plt.hexbin(time_array, vol_array, bins='log')
#plt.title('unequalized EYE diagram with log scale')
#plt.xlabel('time[ps]')
#plt.ylabel('p2p voltage[mV]')
#cb = plt.colorbar()
#cb.set_label('log10(N)')
#plt.savefig('eye_diagram_pre_eq.png')
#
#
#vol  = []
#time = []
#feye = file(  './file_for_eye_post_eq.txt', 'r')
#ready = 0
#i=0
#for line in feye :
#	if(ready!=0) :
#		time.append(((i-bias)%period)/ratio_time)
#		vol.append(float(line)*ratio_volt)
#	if(i==(bias+period-1)) :
#		i=bias
#		ready=ready+1
#	else :
#		i=i+1
#
#time_array = np.array(time)
#vol_array = np.array(vol)
#
#plt.figure(3)
#plt.hexbin(time_array, vol_array, bins='log')
#plt.title('equalized EYE diagram with log scale')
#plt.xlabel('time[ps]')
#plt.ylabel('p2p voltage[mV]')
#cb = plt.colorbar()
#cb.set_label('log10(N)')
#plt.savefig('eye_diagram_post_eq.png')

#ratio_volt = 100
#ready = 0
#for line in feye2 :
#	if(ready!=0) :
#		time.append((i-bias)%period)
#		vol.append(float(line)*ratio_volt)
#	if(i==(bias+period-1)) :
#		i=bias
#		ready=ready+1
#	else :
#		i=i+1
#
#time_array = np.array(time)
#vol_array = np.array(vol)
#
#plt.figure(2)
#plt.hexbin(time_array, vol_array)
#plt.savefig('eye_diagram_post.png')
#plt.show()
#plt.figure(2)
#hist, xedges, yedges = np.histogram2d(time_array, vol_array, bins=400, range=[[0,100],[30,80]])
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#plt.imshow(hist.T, extent=extent, interpolation='nearest', origin='lower')
#plt.colorbar()
#plt.show()
#time_array = np.random.normal(3, 1, 100)
#vol_array  = np.random.normal(1, 1, 100)

#H, xedges, yedges = np.histogram2d(vol_array, time_array, bins=(xedges, yedges))
#fig = plt.figure(figsize=(7, 3))
#ax = fig.add_subplot(131)
#ax.set_title('imshow: equidistant')
#im = plt.imshow(H, interpolation='nearest', origin='low',
#                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#
#ax = fig.add_subplot(132)
#ax.set_title('pcolormesh: exact bin edges')
#X, Y = np.meshgrid(xedges, yedges)
#ax.pcolormesh(X, Y, H)
#ax.set_aspect('equal')
#ax = fig.add_subplot(133)
#ax.set_title('NonUniformImage: interpolated')
#im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
#xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
#ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])
#im.set_data(xcenters, ycenters, H)
#ax.images.append(im)
#ax.set_xlim(xedges[0], xedges[-1])
#ax.set_ylim(yedges[0], yedges[-1])
#ax.set_aspect('equal')
plt.show()

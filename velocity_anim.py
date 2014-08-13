import os,sys
import numpy as np
import processframes
import matplotlib.pyplot as plt
import warpgrid
from scipy.ndimage import interpolation, filters
import motion

reload(warpgrid)
reload(motion)

fs = processframes.FrameSource()

save_dir = "/Users/ethan/insight/anim"

n1 = 2000
n2 = n1+301

z = []
zs = []
zp = []
times = []
for i in range(n1, n2):
	#print i
	zi = fs.get_frame(i)['z']
	ti = fs.get_frame(i)['time_unix']
	zsi = interpolation.zoom(filters.gaussian_filter(zi,1.5),0.25)
	zpi = processframes.patch_out(zsi)
	z.append(zi)
	zs.append(zsi)
	zp.append(zpi)
	times.append(ti)

zs = np.array(zs)
z = np.array(z)
times = np.array(times)

fig = plt.figure(figsize=(5,5))
#ax1 = fig.add_subplot(111)
ax1 = fig.add_axes([0,0,1,1])
#ax2 = fig.add_subplot(122)
plt.show()

c0 = np.zeros((2,)+zs[0].shape)
x0 = np.zeros(32)
extent = (-1,1,-1,1)

for i in range(n2-n1-1):
	print i
	#disp = warpgrid.find_warp(zs[i], zs[i+1], regularize=50)
	#c0 = disp.flatten()
	#vxd, vyd = interpolation.zoom(disp[1],0.2), interpolation.zoom(disp[0],0.2)
	vx, vy, coeffs = motion.displacement_warping(zp[i], zs[i+1], 3, reg_param=0.01, x0=x0)
	x0 = np.copy(coeffs)
	vxd, vyd = interpolation.zoom(vx,0.2), interpolation.zoom(vy,0.2)
	ax1.cla()
	ax1.imshow(z[i+1], origin='lower', vmin=0, vmax=50, extent=extent)
	
	x = np.linspace(extent[0], extent[1], vxd.shape[1])
	y = np.linspace(extent[2], extent[3], vxd.shape[0])
	
	if i > 0:
		decay = 0.3
		vxd_avg = (1 - decay)*vxd_avg + decay*vxd
		vyd_avg = (1 - decay)*vyd_avg + decay*vyd
	else:
		vxd_avg = np.copy(vxd)
		vyd_avg = np.copy(vyd)

	xx, yy = np.meshgrid(x,y)
	ax1.quiver(xx, yy, -vxd_avg, -vyd_avg, color='white', alpha=0.75)

	fname = "%04d.png" % i
	fpath = os.path.join(save_dir, fname)
	fig.savefig(fpath)

	plt.draw()



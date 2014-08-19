import numpy as np


def genmask(shape, rad1=0.15, rad2=1.0):
	rad1 = rad1**2
	rad2 = rad2**2
	x = np.linspace(-1,1,shape[1])
	y = np.linspace(-1,1,shape[0])
	xx,yy= np.meshgrid(x,y)
	r2 = xx**2 + yy**2
	mask = (r2 <= rad2)*(r2 >= rad1)
	return mask

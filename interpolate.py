import numpy as np


def xygrid(x,y, interp, rmax=229):
	nx = len(x)
	ny = len(y)
	z = np.zeros((ny,nx))
	for i, xi in enumerate(x):
		for j, yi in enumerate(y):
			r = np.sqrt(xi**2 + yi**2)
			if (r < rmax):
				az = 90 - np.rad2deg(np.arctan2(yi, xi))
				az = az % 360.
				z[j,i] = interp(r, az)
			else:
				z[j,i] = np.nan
	
	return z
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import optimize
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import map_coordinates
import itertools
from scipy.interpolate import RectBivariateSpline




def gen_mask(shape):
	rad1 = 0.15**2
	rad2 = 1.0**2
	x = np.linspace(-1,1,shape[1])
	y = np.linspace(-1,1,shape[0])
	xx,yy= np.meshgrid(x,y)
	r2 = xx**2 + yy**2
	mask = (r2 <= rad2)*(r2 >= rad1)
	return mask


def sqdiff(im1, im2, shift_x, shift_y, mask):
	height, width = im1.shape
	padding = max(abs(shift_x), abs(shift_y))
	im1p = np.pad(im1, padding, mode='constant')
	im2p = np.zeros(im1p.shape, dtype=im1p.dtype)
	maskp = np.pad(mask, padding, mode='constant')

	im2p[padding+shift_y:padding+shift_y+height, padding+shift_x:padding+shift_x+width] = im2

	delta = (im1p - im2p)**2 * maskp

	return np.sum(delta)



if __name__ == "__main__":
	fpath1 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010020.png'
	#fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010023.png'
	fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010039.png'

	im1b = np.flipud(np.array(Image.open(fpath1), dtype='uint8'))
	im2b = np.flipud(np.array(Image.open(fpath2), dtype='uint8'))

	im1 = np.flipud(np.array(Image.open(fpath1)))*75.0/255.0
	im2 = np.flipud(np.array(Image.open(fpath2)))*75.0/255.0

	m = gen_mask(im1.shape)

	maxshift = 7
	diffmap = np.zeros((maxshift*2+1,)*2)
	shift_x = range(-maxshift, maxshift+1)
	for i, j in itertools.product(range(-maxshift,maxshift+1), repeat=2):
		#print (i,j)
		diffmap[j+maxshift,i+maxshift] = sqdiff(im1, im2, i, j, m)

	s = RectBivariateSpline(shift_x, shift_x, diffmap)
	f = lambda x: s(x[0],x[1])[0,0]
	fprime = lambda x: np.array([s(x[0],x[1],dx=1)[0,0], s(x[0],x[1],dy=1)[0,0]])
	yc, xc = np.unravel_index(diffmap.argmin(), diffmap.shape)
	x0 = [0,0]
	center = optimize.fmin_cg(f, x0, fprime, disp=0)
	shift = -center[::-1]
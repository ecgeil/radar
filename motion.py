import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from PIL import Image
from scipy import optimize
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import map_coordinates
import itertools

fpath1 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010020.png'
#fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010023.png'
fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010030.png'
fpath4 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010039.png'

im1b = np.flipud(np.array(Image.open(fpath1), dtype='uint8'))
im2b = np.flipud(np.array(Image.open(fpath2), dtype='uint8'))

im1 = np.flipud(np.array(Image.open(fpath1)))*75.0/255.0
im2 = np.flipud(np.array(Image.open(fpath2)))*75.0/255.0

im4 = np.flipud(np.array(Image.open(fpath4)))*75.0/255.0

#im1p = np.pad(im1, 10, mode='constant')
#im2p = np.pad(im2, 10, mode='constant')



def gen_mask(shape):
	rad2 = 1.0
	x = np.linspace(-1,1,shape[1])
	y = np.linspace(-1,1,shape[0])
	xx,yy= np.meshgrid(x,y)
	r2 = xx**2 + yy**2
	mask = r2 <= rad2
	return mask




def fill_coords(shape, offset_x, offset_y):
	coords = np.zeros((2,) + shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			coords[:,i,j] = i-offset_y, j-offset_x

	coords[:]

	return coords

def fill_coords2(shape, offset_x, offset_y):
	x = np.arange(shape[1])+ offset_x
	y = np.arange(shape[0])+ offset_y
	coords = np.array(np.meshgrid(x,y)[::-1])

	return coords

zero_offset = fill_coords2(im1.shape, 0, 0)

def coords_c(shape, coeffs_x, coeffs_y):
	#pixel_x = 0.5*nx*(x + 1)
	ny, nx = shape
	x = np.linspace(-1,1,nx)
	y = np.linspace(-1,1,ny)
	offset_x = C.chebgrid2d(x,y,coeffs_x)
	offset_y = C.chebgrid2d(x,y,coeffs_y)
	zero_offset = fill_coords2(shape, 0, 0)
	#zero_offset = np.zeros((2,) + shape)
	coords = zero_offset  + np.array([offset_y, offset_x])
	return coords

def coords_c0(shape, x):
	#pixel_x = 0.5*nx*(x + 1)
	poly_deg = int(len(x)/2)**0.5
	coeffs_x = x[:len(x)/2].reshape((poly_deg,poly_deg))
	coeffs_y = x[len(x)/2:].reshape((poly_deg,poly_deg))
	ny, nx = shape
	x = np.linspace(-1,1,nx)
	y = np.linspace(-1,1,ny)
	offset_x = C.chebgrid2d(x,y,coeffs_x)
	offset_y = C.chebgrid2d(x,y,coeffs_y)
	coords = np.array([offset_y, offset_x])
	return coords


def errfun(x):
	mask = gen_mask(im1.shape)
	coords = fill_coords2(im1.shape, x[0],x[1])
	im3 = map_coordinates(im1, coords)
	diffim = (im2 - im3)
	err = np.sum(np.abs(diffim)*mask)
	return err

def errfun2(x):
	x = np.array(x)
	poly_deg = int(len(x)/2)**0.5
	coeffs_x = x[:len(x)/2].reshape((poly_deg,poly_deg))
	coeffs_y = x[len(x)/2:].reshape((poly_deg,poly_deg))
	coords = coords_c(im1.shape, coeffs_x, coeffs_y)
	im3 = map_coordinates(im1, coords, order=1)
	diffim = (im2 - im3)
	err = np.sum(np.abs(diffim)*mask)/(np.product(im1.shape))
	return err

def errfun_warp(x, im1, im2, mask, reg_param=0.0):
	mask = gen_mask(im1.shape)
	x = np.array(x)
	poly_deg = int(int(len(x)/2)**0.5)
	coeffs_x = x[:len(x)/2].reshape((poly_deg,poly_deg))
	coeffs_y = x[len(x)/2:].reshape((poly_deg,poly_deg))
	coords = coords_c(im1.shape, coeffs_x, coeffs_y)
	im3 = map_coordinates(im1, coords, order=1)
	diffim = (im2 - im3)
	err = np.sum(np.abs(diffim)*mask)/(np.product(im1.shape))

	nx, ny = np.meshgrid(range(poly_deg), range(poly_deg))
	total_deg = nx+ny
	weights = np.tile(total_deg.flatten(), 2)
	penalty = np.sum(weights * np.abs(x))

	return err + reg_param*penalty

def warp_img(im, coeffs):
	x = np.array(coeffs)
	poly_deg = int(int(len(x)/2)**0.5)
	coeffs_x = x[:len(x)/2].reshape((poly_deg,poly_deg))
	coeffs_y = x[len(x)/2:].reshape((poly_deg,poly_deg))
	coords = coords_c(im.shape, coeffs_x, coeffs_y)
	im3 = map_coordinates(im, coords, order=1)

	return im3

def displacement_warping(im1, im2, poly_deg, reg_param=0.0, x0=None):
	mask = gen_mask(im1.shape)
	ncoeffs = 2*poly_deg**2
	if x0 == None:
		x0 = np.zeros(ncoeffs)
	data = dict(im1=im1, im2=im2)
	x1 = optimize.fmin_bfgs(errfun_warp, x0=x0, epsilon=0.01,
			args=(im1, im2, mask, reg_param), disp=False)
	#x1 = optimize.minimize(errfun_warp, x0=x0, method=method, options=dict(eps=0.01),
	#				args=(im1,im2, mask, reg_param))['x']
	c = coords_c0(im1.shape, x1)
	vx, vy = c[1], c[0]
	return (vx, vy, x1)

if __name__ == '__main__' and True:
	vx, vy, coeffs = displacement_warping(im1, im2, 2)

	vx1 = interpolation.zoom(vx, 0.1)
	vy1 = interpolation.zoom(vy, 0.1)

	plt.figure()
	plt.quiver(-vx1, -vy1)
	plt.show()


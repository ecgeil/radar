import numpy as np
from scipy import optimize
from scipy.ndimage import interpolation, filters
from scipy.ndimage.interpolation import map_coordinates






def gen_mask(shape, rad2=0.95):
	x = np.linspace(-1,1,shape[1])
	y = np.linspace(-1,1,shape[0])
	xx,yy= np.meshgrid(x,y)
	r2 = xx**2 + yy**2
	mask = r2 <= rad2
	return mask

def map_img(src, coords):
	coords = np.array(coords)
	coords = coords.reshape((2,) + src.shape)
	offset = coords + np.meshgrid(range(src.shape[0]), range(src.shape[1]), indexing='ij')
	warped = map_coordinates(src, offset, mode='nearest')
	return warped



def errfun(coords, src, target, reg):
	coords = np.array(coords)
	coords = coords.reshape((2,) + src.shape)

	mask  = gen_mask(src.shape)
	
	offset = coords + np.meshgrid(range(src.shape[0]), range(src.shape[1]), indexing='ij')

	warped = map_coordinates(src, offset)

	laplace_sum = np.sum(filters.laplace(coords[0])**2 + filters.laplace(coords[1])**2)

	match_sum = np.sum((warped - target)[mask]**2)

	return match_sum + reg*laplace_sum

def numerical_derivs(coords, src, target, reg, eps=1e-2):
	cshape = coords.shape
	coords=coords.flatten()
	nc = len(coords)
	derivs = np.zeros(nc)
	f0 = errfun(coords, src, target, reg)
	for i in range(nc):
		if (i % 100) == 0:
			print i
		c1 = np.copy(coords)
		c2 = np.copy(coords)
		c1[i] += eps
		c2[i] -= eps
		derivs[i] = (errfun(c1, src, target, reg) - errfun(c2, src,target,reg))/(2*eps)

	return derivs.reshape(cshape)

def errfun_derivs(coords, src, target, reg):
	coords = coords.reshape((2,) + src.shape)
	offset = coords + np.meshgrid(range(src.shape[0]), range(src.shape[1]), indexing='ij')
	warped = map_coordinates(src, offset)
	diff_im = warped - target
	grad_im = np.gradient(src)
	grad_warped = np.zeros((2,) + src.shape)
	laplacian = np.empty(coords.shape)
	laplacian[0] = filters.laplace(coords[0])
	laplacian[1] = filters.laplace(coords[1])
	laplacian_deriv = -11.3*laplacian

	mask  = gen_mask(src.shape)
	for i in range(2):
		grad_warped[i] = map_coordinates(grad_im[i], offset)

	match_deriv = 0.1*4*grad_warped*diff_im*mask

	return (match_deriv + laplacian_deriv*reg).flatten()

def find_warp(src, target, regularize=100, c0=None):
	if c0 == None:
		c0 = np.zeros((2,)+src.shape)

	res = optimize.fmin_cg(errfun, x0=c0, fprime=errfun_derivs, 
		args=(src,target,regularize), disp=False)
	disp = res.reshape((2,) + src.shape)
	return disp


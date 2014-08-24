import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from PIL import Image
from scipy import optimize
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import map_coordinates
import itertools
from utils.genmask import genmask

def coords_c0(shape, x):
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


class Warp:
	def __init__(self, im_shape, poly_deg=3, reg_param=0.0):
		self.ncoeffs = 2*poly_deg**2
		self.poly_deg = poly_deg
		self.shape = im_shape

		self.basis = np.zeros((self.ncoeffs,2) + im_shape )

		for i in range(self.ncoeffs):
			coeffs = np.zeros(self.ncoeffs)
			coeffs[i] = 1.0
			self.basis[i] = coords_c0(self.shape, coeffs)

		self.mask = genmask(self.shape, rad2=0.9)
		self.reg_param = reg_param

		x = np.arange(self.shape[1])
		y = np.arange(self.shape[1])
		self.offset = np.array(np.meshgrid(x,y)[::-1])

		nx, ny = np.meshgrid(range(self.poly_deg), range(self.poly_deg))
		total_deg = nx+ny
		self.reg_weights = self.reg_param * np.tile(total_deg.flatten(), 2)
		self.disp_optmsg = False

		
	def errfun(self, x, im1, im2):
		coords = np.tensordot(x, self.basis, axes=(0,0)) + self.offset
		im3 = map_coordinates(im1, coords, order=1)
		diffim = (im2 - im3)
		#err = np.sum(np.abs(diffim)*mask)/(np.product(im1.shape))
		err = np.sum(diffim**2*self.mask)/(np.product(im1.shape))

		penalty = np.sum(self.reg_weights * np.abs(x))

		return err + penalty

	def errfun_derivs(self, x, im1, im2):
		coords = np.tensordot(x, self.basis, axes=(0,0)) + self.offset
		im3 = map_coordinates(im1, coords, order=1)
		diffim = (im2 - im3)
		gx, gy = np.gradient(im1)
		gx = map_coordinates(gx, coords, order=1)
		gy = map_coordinates(gy, coords, order=1)

		derivs = np.zeros(self.ncoeffs)

		for i in range(self.ncoeffs):
			gradim = gy*self.basis[i,1] + gx*self.basis[i,0]
			derivs[i] = np.sum(-2.0*self.mask*diffim*gradim)/np.prod(im1.shape)

		return derivs + self.reg_weights*np.sign(x)

	def findwarp(self, im1, im2, x0=None):
		if x0 == None:
			x0 = np.zeros(self.ncoeffs)

		x1 = optimize.fmin_bfgs(self.errfun, x0=x0, fprime=self.errfun_derivs,
			args=(im1, im2), disp=self.disp_optmsg)

		c = coords_c0(im1.shape, x1)
		vx, vy = c[1], c[0]
		return (vx, vy, x1)


	def warpim(self, im, coeffs):
		xoff = np.arange(im.shape[1])
		yoff = np.arange(im.shape[0])
		offset = np.array(np.meshgrid(xoff,yoff)[::-1])

		c = np.array(coeffs)
		nc = self.ncoeffs
		pd = self.poly_deg
		coeffs_x = c[:nc/2].reshape((pd,pd))
		coeffs_y = c[nc/2:].reshape((pd,pd))

		ny, nx = im.shape
		x = np.linspace(-1,1,nx)
		y = np.linspace(-1,1,ny)

		disp_x = C.chebgrid2d(x,y,coeffs_x)
		disp_y = C.chebgrid2d(x,y,coeffs_y)

		coords = offset  + np.array([disp_y, disp_x])

		im2 = map_coordinates(im, coords, order=1)
		return im2

	def warpseq(self, im, coeffs, multipliers):
		nout = len(multipliers)
		pred = np.zeros((nout,) + im.shape)
		for i, t in enumerate(multipliers):
			pred[i] =self.warpim(im, coeffs*t)

		return pred





'''Lucas-Kanade warping'''

import numpy as np
from numpy import linalg
from numpy.polynomial import chebyshev as C
from scipy.ndimage.interpolation import map_coordinates
import itertools


def warp(shape, params):
	'''simple shift'''
	ny, nx = shape
	off_x = params[0]
	off_y = params[1]
	x = np.arange(nx)
	y = np.arange(ny)
	xx, yy = np.meshgrid(x+off_x, y+off_y)
	return np.array([yy,xx])

def warp_jac(shape, params):
	'''Jacobian of simple shift
	Return shape (2, len(params), shape[0], shape[1])
	'''
	ny, nx = shape
	off_x = params[0]
	off_y = params[1]
	x0 = np.ones(shape)
	x1 = np.zeros(shape)
	y0 = np.zeros(shape)
	y1 = np.ones(shape)
	return np.array([[x0,x1],[y0, y1]])

def warp_cheb(shape, params, offset=True):
	ny, nx = shape
	params = np.array(params)
	poly_deg = int(len(params)/2)**0.5
	coeffs_x = params[:len(params)/2].reshape((poly_deg,poly_deg))
	coeffs_y = params[len(params)/2:].reshape((poly_deg,poly_deg))
	x = np.linspace(-1,1,nx)
	y = np.linspace(-1,1,ny)
	offset_x = C.chebgrid2d(x,y,coeffs_x)
	offset_y = C.chebgrid2d(x,y,coeffs_y)
	if offset:
		zero_offset = warp(shape, [0,0])
		coords = zero_offset  + np.array([offset_y, offset_x])
	else:
		coords = np.array([offset_y, offset_x])
	return coords

def warp_cheb_jac(shape, params):
	ny, nx = shape
	poly_deg = int(len(params)/2)**0.5
	jac = np.zeros((2,len(params)) + shape)
	for i in range(len(params)):
		coeffs = np.zeros(len(params))
		coeffs[i] = 1.0
		jac[:,i] = warp_cheb(shape, coeffs, offset=False)[::-1]

	return jac
	


def warplk(src, target, warp, warp_jac, p0, mask=None):
	ny, nx = src.shape
	nparam = len(p0)
	grad_im = np.gradient(src)[::-1]
	grad_warped = np.zeros((2,) + src.shape)
	interp_order=1
	p = np.copy(p0)
	p_list = []
	sd_im = np.zeros((nparam,) + src.shape)

	eps = 1e-2
	iter_count = 0
	while iter_count < 20:
		p_list.append(p)
		iter_count += 1
		#print "p: " + str(p)
		coords = warp(src.shape, p)
		src_warped = map_coordinates(src, coords, order=interp_order)
		for i in range(2):
			grad_warped[i] = map_coordinates(grad_im[i], coords, order=interp_order)
		err_im = target - src_warped
		print iter_count, np.sum(np.abs(err_im))
		jac = warp_jac(src.shape, p)

		#compute steepest descent images
		for i in range(nparam):
			sd_im[i] = grad_warped[0]*jac[0,i] + grad_warped[1]*jac[1,i]

		#form hessian
		hess = np.zeros((nparam, nparam))
		if mask != None:
			for iy in range(ny):
				for ix in range(nx):
					hess += np.outer(sd_im[:,iy,ix],sd_im[:,iy,ix])*mask[iy,ix]
		else:	
			for iy in range(ny):
				for ix in range(nx):
					hess += np.outer(sd_im[:,iy,ix],sd_im[:,iy,ix])

		if mask != None:
			errgrad = np.sum(sd_im*err_im*mask, axis=(1,2))
		else:
			errgrad = np.sum(sd_im*err_im, axis=(1,2))
		hess_inv = linalg.inv(hess)

		delta_p = np.dot(hess_inv, errgrad)
		delta_norm = np.linalg.norm(delta_p)
		p = p + delta_p



		#print "delta_p: " + str(delta_p)
		if delta_norm < eps:
			break
	return p

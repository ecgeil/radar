import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import optimize
from numpy.polynomial import chebyshev as C
import itertools
from numpy import linalg


def subimage(im, xc,yc, w, h, return_offset=False):
	x1 = xc-w/2
	x2 = xc+w/2+1
	y1 = yc-h/2
	y2 = yc+h/2+1
	if return_offset:
		return im[y1:y2,x1:x2], x1, y1
	else:
		return im[y1:y2,x1:x2]


def local_match(im_templ, im_search, xc, yc, w_templ, w_search):
	template = subimage(im_templ, xc, yc, w_templ, w_templ)
	searchimg = subimage(im_search, xc, yc, w_search, w_search)
	score = cv2.matchTemplate(searchimg, template, method=cv2.TM_SQDIFF_NORMED)
	return score

def subpixel_peak(z):
	w = 3
	yc, xc = np.unravel_index(z.argmin(), z.shape)

	neighborhood, offx, offy = subimage(z, xc, yc, w, w, return_offset=True)
	if min(neighborhood.shape) < w: #edge pixel
		#return np.array([xc,yc])
		return np.array([z.shape[0]/2]*2)

	x = np.arange(w)
	s = RectBivariateSpline(x,x,neighborhood, kx=2, ky=2)
	f = lambda x: s(x[0],x[1])[0,0]
	fprime = lambda x: np.array([s(x[0],x[1],dx=1)[0,0], s(x[0],x[1],dy=1)[0,0]])
	x0 = [0.5*(w-1), 0.5*(w-1)]
	center = optimize.fmin_cg(f, x0, fprime, disp=0)
	center = np.array(center)
	return center + np.array([offx, offy])



def local_offset(current, prev, xc, yc, w_curr, w_prev):
	score = local_match(current, prev, xc, yc, w_curr, w_prev)
	peak_pos = subpixel_peak(score)
	center = np.array([0.5*(score.shape[0] - 1)]*2)

	return (peak_pos - center, np.ptp(score))

def grid_offset(current, prev, grid_points, w_curr, w_prev):
	off_grid = np.zeros((len(grid_points),)*2 + (2,))
	quality_grid = np.zeros((len(grid_points),)*2)
	for i, xi in enumerate(grid_points):
		for j, yj in enumerate(grid_points):
			off, quality = local_offset(im2b, im1b, xi, yj, w_curr, w_prev)
			off_grid[j,i] = off
			quality_grid[j,i] = quality

	return off_grid, quality_grid

def fit_field(current, prev, w_curr, w_prev, degree):
		grid_points= np.arange(25,175.1,5)
		off_grid, quality_grid = grid_offset(current, prev, grid_points, 15, 27)

def chebvals(shape, degree): 
	"""2d grid of chebyshev polynomials"""
	nd = degree+1
	vals = np.zeros(shape + (nd**2,))
	coeffs = np.zeros((nd,nd))
	x = np.linspace(-1,1, shape[1])
	y = np.linspace(-1,1, shape[0])
	for i,j in itertools.product(range(degree+1), repeat=2):
		idx = nd*i + j
		coeffs[i,j] = 1.0
		vals[...,idx] = C.chebgrid2d(x,y,coeffs)
		coeffs[i,j] = 0.0
	return vals

if __name__ == '__main__' and True:
	fpath1 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010020.png'
	fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010033.png'

	im1b = np.flipud(np.array(Image.open(fpath1), dtype='uint8'))
	im2b = np.flipud(np.array(Image.open(fpath2), dtype='uint8'))

	im1 = np.flipud(np.array(Image.open(fpath1)))*75.0/255.0
	im2 = np.flipud(np.array(Image.open(fpath2)))*75.0/255.0

	grid_points= np.arange(25,175.1,5)
	off_grid, quality_grid = grid_offset(im1b, im2b, grid_points, 15,27)

	interp_x = RectBivariateSpline(grid_points, grid_points, off_grid[...,0])
	interp_y = RectBivariateSpline(grid_points, grid_points, off_grid[...,1])
	interp_q = RectBivariateSpline(grid_points, grid_points, quality_grid)

	interp_points_x = np.arange(im1.shape[1])
	interp_points_y = np.arange(im1.shape[0])

	vx = interp_x(interp_points_x, interp_points_y)
	vy = interp_y(interp_points_x, interp_points_y)
	qi = interp_q(interp_points_x, interp_points_y)
	
	x_coords = np.linspace(-1,1,im1.shape[1])
	y_coords = np.linspace(-1,1,im1.shape[0])
	threshold = 40
	valid_pts = np.nonzero((im1b > threshold) & (qi > 0.2))

	cv = chebvals(im1.shape, 0)
	A = cv[valid_pts]
	b = vx[valid_pts]
	coeffs_x, resid, rank, singvals = linalg.lstsq(A, b)
	coeffs_y, resid, rank, singvals = linalg.lstsq(A, vy[valid_pts])

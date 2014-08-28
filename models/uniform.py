"""Prediction by extrapolating along mean motion vector"""

import numpy as np
from numpy.random import normal
from numpy.random import standard_normal
import matplotlib.pyplot as plt
from PIL import Image
from scipy import optimize
from scipy.ndimage import interpolation, filters
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import map_coordinates
import itertools
from scipy.interpolate import RectBivariateSpline
from skimage import restoration
import predictor


def gen_mask(shape, rad1=0.15, rad2=0.95):
	rad1 = rad1**2
	rad2 = rad2**2
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

def shift_indices(n, shift):
	"""Find indices for copying one image onto another,
	with a shift"""
	if shift >= 0: #shifting right
		xi1 = 0
		xi2 = n - shift
		xf1 = shift
		xf2 = n
	else:
		xi1 = -shift
		xi2 = n
		xf1 = 0
		xf2 = n + shift
	return (xi1, xi2, xf1, xf2)



def findshift(im1, im2, maxshift=8):
	"""Compute shift from im1 to im2"""
	m = gen_mask(im1.shape)
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
	return shift

def shiftim(im, shift_x, shift_y):
	"""Offset image by shift_x, shift_y"""
	ny, nx = im.shape
	newim = np.zeros(im.shape, dtype=im.dtype)
	xi1, xi2, xf1, xf2 = shift_indices(nx, shift_x)
	yi1, yi2, yf1, yf2 = shift_indices(ny, shift_y)
	newim[yf1:yf2,xf1:xf2] = im[yi1:yi2,xi1:xi2]
	return newim

def find_velocity(times, frames, maxshift=10):
	frame_times = np.array(times)
	nframes = len(frames)
	last_idx = np.argmax(frame_times)
	velocities = []
	npairs = nframes*(nframes-1)/2
	denom = 0
	maxshift = 10
	for i in range(nframes):
		for j in range(i+1, nframes):
			dt = frame_times[i] - frame_times[j]
			offset = findshift(frames[i], frames[j], maxshift)
			if abs(max(offset)) > maxshift:
				continue
			denom += 1
			print (i, j, offset, dt)
			velocity = -offset/dt
			velocities.append(velocity)

	denom = min(1, denom-1)
	velocities = np.array(velocities)
	(vx1, vy1) = velocities[:,0].mean(), velocities[:,1].mean()
	vx = np.mean(velocities[:,0])
	vy = np.mean(velocities[:,1])
	sx = np.std(velocities[:,0])/denom + 0.2*abs(vx) + 5e-4
	sy = np.std(velocities[:,1])/denom + 0.2*abs(vy) + 5e-4
	return (vx, vy, sx, sy)

def extrapolate_prob(im, vx, vy, sx, sy, output_times, threshold, num_trials=100):
	"""velocities vx, vy per frame
		standard deviations sx, sy
		output_times are times after the input frame time
	"""
	num_frames = len(output_times)
	prob = np.zeros((num_frames,)+ im.shape)
	for i in range(num_frames):
		
		spread_x = sx * output_times[i]
		spread_y = sy * output_times[i]
		for j in range(num_trials):
			ti = output_times[i]
			off_x = ti*(vx  +  sx*standard_normal())
			off_y = ti*(vy  +  sx*standard_normal())

			off_x = int(round(off_x))
			off_y = int(round(off_y))

			if abs(off_x) < 0.5*im.shape[1] and abs(off_y) < 0.5*im.shape[0]:
				shifted = shiftim(im, off_x, off_y)
			else:
				shifted = np.zeros(im.shape)
			prob[i] += shifted > threshold


	return np.clip(prob*(1.0/num_trials), 0.0, 1.0)

def extrapolate(im, vx, vy, sx, sy, output_times, num_trials=100):
	"""velocities vx, vy per frame
		standard deviations sx, sy
		output_times are times after the input frame time
	"""
	num_frames = len(output_times)
	prob = np.zeros((num_frames,)+ im.shape)
	for i in range(num_frames):
		
		spread_x = sx * output_times[i]
		spread_y = sy * output_times[i]
		for j in range(num_trials):
			ti = output_times[i]
			off_x = ti*(vx  +  sx*standard_normal())
			off_y = ti*(vy  +  sx*standard_normal())

			off_x = int(round(off_x))
			off_y = int(round(off_y))

			if abs(off_x) < 0.5*im.shape[1] and abs(off_y) < 0.5*im.shape[0]:
				shifted = shiftim(im, off_x, off_y)
			else:
				shifted = np.zeros(im.shape) 

			
			prob[i] += shifted


	return prob*(1.0/num_trials)

class UniformVelocityPredictor(predictor.Predictor):
	def predict_prob(self, times, frames, output_times, threshold=20):
		last_idx = np.argmax(times)
		last_frame = frames[last_idx]
		vx, vy, sx, sy = find_velocity(times, frames)
		print (vx, vy, sx, sy)
		output_times = np.array(output_times)
		output_times -= times[last_idx]
		prob = extrapolate_prob(last_frame, vx, vy, sx, sy, output_times, threshold)
		return prob

	def predict(self, times, frames, output_times):
		last_idx = np.argmax(times)
		last_frame = frames[last_idx]
		vx, vy, sx, sy = find_velocity(times, frames)
		print (vx, vy, sx, sy)
		output_times = np.array(output_times)
		output_times -= times[last_idx]
		pred = extrapolate(last_frame, vx, vy, sx, sy, output_times)
		return pred


		
import numpy as np
from numpy.random import normal
from numpy.random import standard_normal
import matplotlib.pyplot as plt
from PIL import Image
from scipy import optimize
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.interpolation import map_coordinates
import itertools
from scipy.interpolate import RectBivariateSpline
import predictor



def gen_mask(shape, rad1=0.15, rad2=1.0):
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

def shift_indices(n, shift):
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


def shiftim(im, shift_x, shift_y):
	"""Offset image by shift_x, shift_y"""
	ny, nx = im.shape
	newim = np.zeros(im.shape, dtype=im.dtype)
	xi1, xi2, xf1, xf2 = shift_indices(nx, shift_x)
	yi1, yi2, yf1, yf2 = shift_indices(ny, shift_y)
	newim[yf1:yf2,xf1:xf2] = im[yi1:yi2,xi1:xi2]
	return newim

def shift_series(im, offsets):
	"""generate a stack of images at successive offsets"""
	noff = len(offsets)
	stack = np.zeros((noff,) + im.shape)
	for i in range(noff):
		stack[i] = shiftim(im, int(offsets[i][0]), int(offsets[i][1]))
	return stack

def accumulate(im, vx, vy, sx, sy, output_times, num_trials=10, threshold=20):
	"""velocities mx, my per frame
		standard deviations sx, sy
	"""
	num_frames = len(output_times)
	imt = np.where(im > threshold, im/threshold, 0)
	accum = np.zeros((num_frames,)+ im.shape)
	for i in range(num_trials):
		vxi = vx + sx*standard_normal()
		vyi = vy + sy*standard_normal()
		offsets = [[vxi*t, vyi*t] for t in output_times]
		accum += shift_series(imt, offsets)

	return np.clip(accum*(1.0/num_trials), 0.0, 1.0)

def predict_prob(frames, frame_times, output_times):
	"""
		frames: successive dBz images on rectangular grid
		frame_times: times in some unit (e.g. seconds)
		nout: number of output frames
		interval: time between output frames

		return: predicted probability on grid of the same shape as frames_z
	"""
	frame_times = np.array(frame_times)
	nframes = len(frames)
	last_idx = np.argmax(frame_times)
	velocities = []
	npairs = nframes*(nframes-1)/2
	denom = npairs-1 if npairs > 1 else 1.0
	maxshift = 12
	for i in range(nframes):
		for j in range(i+1, nframes):
			dt = frame_times[j] - frame_times[i]
			offset = findshift(frames[i], frames[j], maxshift)
			if abs(max(offset)) > maxshift:
				continue
			print (i, j, offset, dt)
			velocity = offset/dt
			print velocity
			velocities.append(velocity)
	velocities = np.array(velocities)
	(vx1, vy1) = velocities[:,0].mean(), velocities[:,1].mean()
	
	vx = np.mean(velocities[:,0])
	vy = np.mean(velocities[:,1])
	sx = np.std(velocities[:,0])/denom + 0.2*abs(vx) + 0.002
	sy = np.std(velocities[:,1])/denom + 0.2*abs(vy) + 0.002

	print (vx, vy, sx, sy)

	prob = accumulate(frames[last_idx], vx, vy, 1.5*sx, 1.5*sy, output_times,
						num_trials=40)

	return prob




class UniformPredictor(predictor.Predictor):
	def predict_prob(self, times, frames, output_times, threshold=20):
		prob = predict_prob(frames, times, output_times)
		return prob

	def predict(self, times, frames, output_times):
		last_idx = np.argmax(times)
		frame = frames[last_idx]
		return [frame]*len(output_times)



if __name__ == "__main__":
	fpath1 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010020.png'
	fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010023.png'
	fpath3 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010030.png'
	fpath4 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010039.png'

	im1b = np.flipud(np.array(Image.open(fpath1), dtype='uint8'))
	im2b = np.flipud(np.array(Image.open(fpath2), dtype='uint8'))

	im1 = np.flipud(np.array(Image.open(fpath1)))*75.0/255.0
	im2 = np.flipud(np.array(Image.open(fpath2)))*75.0/255.0
	im3 = np.flipud(np.array(Image.open(fpath3)))*75.0/255.0
	im4 = np.flipud(np.array(Image.open(fpath4)))*75.0/255.0


	prob = predict_prob([im1, im2, im3], [20.,23., 30], 10, 3.0)
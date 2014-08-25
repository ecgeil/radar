import numpy as np
from scipy.ndimage import morphology
from scipy.ndimage import filters
import skimage.morphology as skmorph


def distance_trend(times, frames, threshold=25, min_size=12):
	nf = len(frames)
	zt = filters.gaussian_filter(frames,1.5) > threshold
	d_outer = np.zeros((nf,) + frames[0].shape)
	d_inner = np.zeros((nf,) + frames[0].shape)

	for i in range(nf):
		skmorph.remove_small_objects(zt, min_size=min_size, in_place=True)
		d_outer[i] = morphology.distance_transform_edt(np.invert(zt[i]))
		d_inner[i] = morphology.distance_transform_edt(zt[i])

	return d_outer, d_inner

class DistanceOuter:
	def __init__(self, threshold=20, min_size=12):
		self.thresh = threshold
		self.min_size = min_size

	def predict(self, times, frames, output_times):
		nout = len(output_times)
		nf = len(frames)

		last_idx = np.argmax(times)
		zt = frames[last_idx] > self.thresh
		skmorph.remove_small_objects(zt, min_size=self.min_size, in_place=True)
		d = morphology.distance_transform_edt(np.invert(zt))
		return np.tile(d, (nout,1,1))


class DistanceInner:
	def __init__(self, threshold=20, min_size=12):
		self.thresh = threshold
		self.min_size = min_size

	def predict(self, times, frames, output_times):
		nout = len(output_times)
		nf = len(frames)

		last_idx = np.argmax(times)
		zt = frames[last_idx] < self.thresh
		skmorph.remove_small_objects(zt, min_size=self.min_size, in_place=True)
		d = morphology.distance_transform_edt(np.invert(zt))
		return np.tile(d, (nout,1,1))
import numpy as np
from scipy.ndimage import filters

import predictor


class DiffusionPredictor(predictor.Predictor):
	def __init__(self, diffusion_constant=5e-3):
		self.dc = diffusion_constant #km/s



	def predict(self, times, frames, output_times):
		nout = len(output_times)
		last_idx = np.argmax(times)
		frame = frames[last_idx]
		pred = np.zeros((nout,) + frame.shape)
		dc_pix = self.dc * 460.0/frame.shape[0]
		output_dt = np.array(output_times) - times[last_idx]
		for i in range(nout):
			sigma = dc_pix * output_dt[i]
			sigma = min(sigma, frame.shape[0])
			#print "sigma = %f" % sigma
			pred[i] = filters.gaussian_filter(frame, sigma)

		return pred

	def predict_prob(self, times, frames, output_times, threshold=20):
		nout = len(output_times)
		last_idx = np.argmax(times)
		frame = frames[last_idx] > threshold
		pred = np.zeros((nout,) + frame.shape)
		dc_pix = self.dc * 460.0/frame.shape[0]
		output_dt = np.array(output_times) - times[last_idx]
		for i in range(nout):
			sigma = dc_pix * output_dt[i]
			sigma = min(sigma, frame.shape[0])
			pred[i] = filters.gaussian_filter(frame, sigma)

		return pred
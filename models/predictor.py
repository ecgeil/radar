import numpy as np

class Predictor():
	"""frames: sequence of 2d arrays of reflectivity in dBZ
	   times: the times at which the frames were recorded
	   output_times: the future times at which to predict
	   threshold: threshold for rain (20 dBZ default)
	   return: either a probability or a predicted reflectivity
	   			on the same grid as the input frames
	"""
	def __init__(self):
		pass

	def predict_prob(self, times, frames, output_times, threshold=20):
		last_idx = np.argmax(times)
		last_frame = frames[last_idx]
		prob = np.where(last_frame > threshold, 1.0, 0.0)
		return [prob]*len(output_times)

	def predict(self, times, frames, output_times):
		last_idx = np.argmax(times)
		frame = frames[last_idx]
		return [frame]*len(output_times)




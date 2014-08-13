import numpy as np

class Predictor():
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




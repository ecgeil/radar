import numpy as np


class LogisticPredictor():
	def __init__(self, coeffs):
		self.coeffs = coeffs


	def predict(self,x):
		x = np.atleast_2d(x)
		s = np.sum(self.coeffs[:-1]*np.array(x), axis=1) + self.coeffs[-1]
		return 1.0/(1 + np.exp(-s))

	
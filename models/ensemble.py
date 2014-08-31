import numpy as np
from scipy.interpolate import interp1d
import logging



def logprob(coeffs,x):
	'''coeffs: 1d array of length nvar
		x: nd array, (nobs, nvar)
		returns: predicted probability, 1d array (nobs,)
	'''
	x = np.atleast_2d(x)
	s = np.sum(coeffs[:-1]*np.array(x), axis=1) + coeffs[-1]
	return 1.0/(1 + np.exp(-s))

class LogisticPredictor():
	def __init__(self, coeffs):
		self.coeffs = coeffs


	def predict(self,x):
		x = np.atleast_2d(x)
		s = np.sum(self.coeffs[:-1]*np.array(x), axis=1) + self.coeffs[-1]
		return 1.0/(1 + np.exp(-s))

	
class Ensemble():
	def __init__(self, models, coeffs, times):
		'''
		models: list of models, which have a predict() method
		coeffs: regression coefficients (len(times), len(models)+1)
			in same order as models, with the intercepts in the last column
		times: the times for which the coefficients were computed
		'''
		self.models = models
		self.nmod = len(models)
		self.coeffs = coeffs
		self.ctimes = times
		self.interpolator = interp1d(times, coeffs, axis=0)

	def predict(self, times, frames, output_times):
		output_times = np.array(output_times)
		times = np.array(times)
		delta_times = output_times - np.max(times)
		logging.debug("input frame ages: %s", str(delta_times))
		if (np.max(delta_times) > np.max(self.ctimes) or 
			np.min(delta_times) < np.min(self.ctimes)):
			raise ValueError("prediction time is outside model calibration times")
		fshape = frames[0].shape
		nout = len(output_times)
		logging.info( "nout: %d",  nout)
		

		pred = np.zeros((nout,) + fshape + (self.nmod,) )
		for i, m in enumerate(self.models):
			pred[...,i] = m.predict(times, frames, output_times)

		prob = np.zeros((nout,) + fshape)
		pshape = (np.product(fshape), self.nmod)
		for i, ti in enumerate(delta_times):
			coeffs_interp = self.interpolator(ti)
			prob_i = logprob(coeffs_interp, pred[i].reshape(pshape))
			prob[i] = prob_i.reshape(fshape)
		return prob




import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import local_prediction
import localpred2
from data import nexradutils
from scipy.ndimage import filters
import time
from pyproj import Proj
import logging

logit = lambda x: 1.0/(1+np.exp(-x))

def makeplot(lon, lat, fpath, station='kdix'):
	times, prob, pred = localpred2.point_prediction(station, 
					nframes=20, interval=180, lon=lon,lat=lat)

	try:
		if pred:
			fig = plt.figure(figsize=(6,4))
			ax = fig.add_subplot(111)

			prob2 = logit(6.583*(prob - 0.5))
			times_i = np.linspace(times[0], times[-1], 100)
			prob2_i = np.interp(times_i, times, prob2)
			prob2_f = filters.gaussian_filter1d(prob2_i, 3)

			ax.plot(times_i, prob2_f, lw=2)
			ax.set_xlim(0,50)
			ax.set_ylim(0,1.0)
			ax.set_xlabel('time (min)')
			ax.set_ylabel('probability')
			fig.tight_layout()
			fig.savefig(fpath, transparent=True, dpi=200)

			utmzone = pred['utmzone']

			p = Proj(proj='utm', zone=utmzone, ellps='WGS84')
			x, y = p(lon, lat)
			extent = pred['extent']

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.imshow(pred['prob'][0], origin='lower', extent=extent)
			#ax.set_xlim((extent[0], extent[1]))
			#ax.set_xlim((extent[2], extent[3]))
			ax.plot([x], [y], 'o')
			fig.savefig("pred0.png")

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.imshow(pred['prob'][-1], origin='lower', extent=extent)
			ax.plot([x], [y], 'o')
			fig.savefig("pred1.png")

			logging.info( "figure saved at %s", fpath)
		else:
			logging.error("Couldn't get prediction")

			fig = plt.figure(figsize=(6,4))
			ax = fig.add_subplot(111)
			ax.plot([0,1],[0,1], 'r')
			ax.text(0.5,0.5, "?", fontsize=72)
			fig.savefig(fpath, transparent=True, dpi=200)
	except Exception, e:
		logging.exception(e)
	finally:
		pass


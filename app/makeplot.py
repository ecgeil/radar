import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import local_prediction
import nexradutils
import time



def makeplot(x, y, fpath, station='kdix'):
	times, prob, pred = local_prediction.specific_prediction(station, nframes=22, interval=180, lon=x,lat=y)
	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(111)
	ax.plot(times, prob, lw=2)
	ax.set_xlim(0,60)
	ax.set_ylim(0,1.0)
	ax.set_xlabel('time (min)')
	ax.set_ylabel('probability')
	fig.tight_layout()
	fig.savefig(fpath, transparent=True, dpi=200)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(pred['prob'][0], origin='lower')
	fig.savefig("pred0.png")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(pred['prob'][-1], origin='lower')
	fig.savefig("pred1.png")

	print "figure saved at " + fpath


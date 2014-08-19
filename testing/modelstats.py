import numpy as np

from data import processframes
from utils.genmask import genmask



def test_prob(predictor, npast=5, nfuture=12, limit=100):
		fs = processframes.FrameSource()

		mask = genmask(fs.shape, rad1=0.05, rad2=0.8)
		mask_els = np.sum(mask)
		threshold = 25
		ntotal = nfuture + npast
		
		z = np.zeros((ntotal,) + fs.shape)
		times = np.zeros(ntotal)
		pred_times = []
		pred_scores = []
		zmean = []


		start_idx = 0


		for start_idx in range(0, limit, ntotal):
			
			f1 = fs.get_frame(start_idx)
			
			rain_area = 1.0*np.sum(f1['z'] > threshold) / mask_els
			print start_idx, rain_area
			if rain_area < 0.03:
				print "skipping"
				continue

			now_idx = npast - 1
			for i, frame in enumerate(fs.iter_frames(start_idx, start_idx+ntotal)):
				z[i] = frame['z']
				times[i] = frame['time_unix']


			times -= times[now_idx]
			

			pred = predictor.predict_prob(times[:now_idx+1], z[:now_idx+1], times[now_idx:])

			for i in range(nfuture):
				zidx = now_idx + i
	
				p_act = z[zidx] > threshold
				cc = np.sum((p_act[mask] - pred[i][mask])**2)/mask_els
				

				pred_times.append(times[zidx])
				pred_scores.append(cc)
				zmean.append(rain_area)
				

		return pred_times, pred_scores


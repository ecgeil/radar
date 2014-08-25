import numpy as np

from data import processframes
from utils.genmask import genmask
from skimage.restoration import denoise_bilateral


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

def interp_frames(times, z, outtimes):
	nout = len(outtimes)
	zinterp = np.zeros((nout,) + z[0].shape, dtype=z.dtype)

	for i in range(nout):
		tinterp = outtimes[i]
		idx = np.max(np.argwhere(times < tinterp))
		t1 = times[idx]
		t2 = times[idx+1]
		r = (tinterp - t1)/float(t2 - t1)

		zinterp[i] = (1-r)*z[idx] + r*z[idx+1]

	return zinterp


def random_sample(predictor, npast=5, nsamp=10):
		fs = processframes.FrameSource()
		nfuture = 25
		mask = genmask(fs.shape, rad1=0.05, rad2=0.8)
		mask_els = np.sum(mask)
		threshold = 25
		ntotal = nfuture + npast
		
		z = np.zeros((ntotal,) + fs.shape)
		times = np.zeros(ntotal)
		pred_times = np.arange(0,3601,300)
		npred = len(pred_times)

		pred_scores = np.zeros(npred)

		prob_bins = np.arange(0.1,1.01,0.1)
		prob_bincounts = np.zeros((npred, len(prob_bins) + 1))
		prob_trialcounts = np.zeros((npred, len(prob_bins) + 1))

		x = [[]]*npred
		y = [[]]*npred

		rand_score = 0

		start_idx = 0
		rain_sum = 0

		trial = 0
		while trial < nsamp:
			start_idx = np.random.randint(0, fs.num_frames - 50)
			
			f1 = fs.get_frame(start_idx)
			
			rain_area = 1.0*np.sum(f1['z'] > threshold) / mask_els
			rain_sum += rain_area
			print start_idx, rain_area
			if rain_area < 0.02:
				print "skipping"
				continue

			trial += 1
			print "Trial %d of %d" % (trial, nsamp)
			rain_sum += rain_area

			now_idx = npast - 1
			for i, frame in enumerate(fs.iter_frames(start_idx, start_idx+ntotal)):
				times[i] = frame['time_unix']
				z[i] = frame['z']

			times -= times[now_idx]

			zinterp = interp_frames(times, z, pred_times)
			zinterp[0] = z[now_idx]


			pred = predictor.predict_prob(times[:now_idx+1], z[:now_idx+1], pred_times)

			frac = 0.1
			for i in range(npred):
				p_act = zinterp[i] > threshold
				sample = (np.random.random(fs.shape) > (1 - frac))*mask
				print np.sum(sample)
				cc = np.sum((p_act[mask] - pred[i][mask])**2)/mask_els
				#cc = np.sum((zinterp[i][mask] - pred[i][mask])**2)/mask_els
				#cc = np.corrcoef(zinterp[i][mask], pred[i][mask])[0,1]
				cc_rand = np.sum((0.147 - pred[i][mask])**2)/mask_els

				bin_indices = np.digitize(pred[i][mask], prob_bins)
				pmi = p_act[mask]
				
				for j in range(mask_els):
					prob_bincounts[i][bin_indices[j]] += pmi[j]
					prob_trialcounts[i][bin_indices[j]] += 1

				sample_els = np.sum(sample)
				for j in range(sample_els):
					x[i].append(pred[i][sample][j])
					y[i].append(p_act[sample][j])

				rand_score += cc_rand

				pred_scores[i] += cc
				
		print "rain fraction: %f" % (rain_sum/float(nsamp))
		print "random skill: %f" % (rand_score/nsamp)
		pred_scores *= 1.0/nsamp

		data = {}
		data['pred_times'] = pred_times
		data['pred_scores'] = pred_scores
		data['prob_bins'] = prob_bins
		data['prob_bincounts'] = prob_bincounts
		data['prob_trialcounts'] = prob_trialcounts
		data['prob_fractions'] = 1.0*prob_bincounts/prob_trialcounts
		data['zinterp'] = zinterp
		data['pred'] = pred
		data['mask_els'] = mask_els
		data['x'] = x
		data['y'] = y
		return data

def random_sample_multi(predictors, npast=4, maxpts=1000):
		print "here"
		fs = processframes.FrameSource()
		nfuture = 25
		mask = genmask(fs.shape, rad1=0.05, rad2=0.8)
		mask_els = np.sum(mask)
		threshold = 20
		ntotal = nfuture + npast
		pts_count = 0
		
		z = np.zeros((ntotal,) + fs.shape)
		times = np.zeros(ntotal)
		pred_times = np.arange(0,3601,300)
		num_pred_times = len(pred_times)


		x = np.zeros((num_pred_times, maxpts, len(predictors)))
		y = np.zeros((num_pred_times, maxpts))

		start_idx = 0
		rain_sum = 0

		trial = 0
		while pts_count < maxpts:
			start_idx = np.random.randint(0, fs.num_frames - 50)
			
			f1 = fs.get_frame(start_idx)
			
			rain_area = 1.0*np.sum(f1['z'] > threshold) / mask_els
			rain_sum += rain_area
			print start_idx, rain_area
			if rain_area < 0.02:
				print "skipping"
				continue

			trial += 1
			print "Trial %d" % (trial)
			print "Point %d of %d" % (pts_count, maxpts)
			rain_sum += rain_area

			now_idx = npast - 1
			for i, frame in enumerate(fs.iter_frames(start_idx, start_idx+ntotal)):
				z[i] = frame['z']
				times[i] = frame['time_unix']

			times -= times[now_idx]

			zinterp = interp_frames(times, z, pred_times)
			zinterp[0] = z[now_idx]

			preds = []
			for predictor in predictors:
				pred = predictor.predict(times[:now_idx+1], z[:now_idx+1], pred_times)
				
				preds.append(pred)

			frac = 0.01
			#p_act = zinterp > threshold
			p_act = zinterp
			sample = (np.random.random(fs.shape) > (1 - frac))*mask
			

			sample_els = np.sum(sample)
			print "sample size: %d" % sample_els

			for j in range(sample_els):
				if pts_count >= maxpts:
					break
				for i in range(num_pred_times):
					for k, pred in enumerate(preds):
						
						x[i,pts_count,k] = pred[i][sample][j]
					y[i,pts_count] = p_act[i][sample][j]

				pts_count += 1

		data = {}
		data['pred_times'] = np.array(pred_times)
		data['zinterp'] = zinterp
		data['preds'] = np.array(preds)
		data['mask_els'] = mask_els
		data['x'] = x
		data['y'] = y
		return data







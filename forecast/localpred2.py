import numpy as np
from data import nexradutils
from data import radardb
from models import distance, diffusion, warp, ensemble



coeffs = np.array([[  2.46491812e+00,  -2.02737506e-02,   2.85547678e-01,
          2.85547678e-01,  -1.39527233e+01],
       [  3.41048144e-01,  -4.09643933e-02,  -3.51311605e-02,
          3.62890646e-01,  -6.99988429e+00],
       [  2.34887549e-01,  -4.89958945e-02,   1.95213165e-02,
          2.50584833e-01,  -5.34725960e+00],
       [  7.03256067e-02,  -5.97254990e-02,   3.12970330e-02,
          2.18664706e-01,  -4.64474257e+00],
       [  5.56821692e-02,  -7.27768727e-02,   2.55712733e-02,
          1.84873412e-01,  -4.02475441e+00],
       [  3.06452449e-02,  -7.50116408e-02,   3.18100129e-02,
          1.69653217e-01,  -3.72391426e+00],
       [  2.42346564e-02,  -7.80518125e-02,   3.03926490e-02,
          1.58475456e-01,  -3.50457617e+00],
       [  7.23178521e-03,  -7.97552035e-02,   3.14786899e-02,
          1.59158120e-01,  -3.37700373e+00],
       [  1.60770194e-02,  -7.58334185e-02,   3.98534406e-02,
          1.50150442e-01,  -3.31802134e+00],
       [ -1.16156646e-02,  -7.53754850e-02,   4.56853291e-02,
          1.40720056e-01,  -3.22378996e+00],
       [ -9.73400430e-03,  -7.76278840e-02,   4.41322512e-02,
          1.39176398e-01,  -3.09982741e+00],
       [ -9.91378703e-03,  -7.21481621e-02,   6.03463283e-02,
          1.31741939e-01,  -3.09917718e+00],
       [  2.66259064e-03,  -7.49638672e-02,   5.52444043e-02,
          1.19719031e-01,  -2.93148183e+00]])

coeff_times = np.array([   0,  300,  600,  900, 1200, 1500, 1800, 2100, 2400, 2700, 3000,
       3300, 3600])


def station_prediction(station, nframes=12, interval=300, prev_frames=4):
	frames = radardb.get_latest(station, prev_frames)
	z = np.array([f['z'] for f in frames])
	frame_times = np.array([f['unix_time'] for f in frames])

	#Sort in ascending time order
	sa = np.argsort(frame_times)
	frame_times = frame_times[sa]
	z = z[sa]

	mods = [distance.DistanceInner(), 
			distance.DistanceOuter(), 
			diffusion.DiffusionPredictor(), 
			warp.Warp((50,50), poly_deg=4, reg_param=0.01)]

	ensemble_mod = ensemble.Ensemble(mods, coeffs, coeff_times)

	output_times = np.array([1.0*interval*i for i in range(nframes)])
	rel_times = frame_times - frame_times[-1]
	print rel_times
	prob = ensemble_mod.predict(rel_times, z, output_times)

	pred = {}
	pred['prob'] = prob
	pred['prev_z'] = z
	pred['prev_t'] = rel_times
	pred['extent'] = frames[0]['extent']
	pred['interval'] = interval
	pred['start_time'] = frame_times[-1]

	return pred

def point_prediction(station='kddc', nframes=20, interval=180, lon=-97.9297, lat=38.0608):
	x, y, utmzone = nexradutils.lonlat2utm(lon, lat)

	pred = station_prediction(station, nframes, interval)
	gridsize = pred['prob'][0].shape[0]
	print gridsize
	print (x,y,utmzone)
	print pred['extent']
	px, py = nexradutils.pix4coord(x, y, gridsize, extent=pred['extent'])
	print (px, py)
	trace = pred['prob'][:,py,px]
	
	times = [interval/60. * i for i in range(nframes)]
	return times, trace, pred
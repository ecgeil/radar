import netCDF4
import numpy as np
from ftplib import FTP
import subprocess
from subprocess import call
import os
from datetime import datetime
from data import nexradutils
from data import radardb
from models import uniform
import pyproj

import time
import traceback

import matplotlib.pyplot as plt

debug_frames = []

host = 'tgftp.nws.noaa.gov'
ftppath = 'SL.us008001/DF.of/DC.radar/DS.p19r0/'

def download_latest(station, local_dir, timelimit=1800):
	cache=False
	dirname = os.path.join(ftppath, "SI."+station)
	print "retrieving files from " + dirname

	try:
		ftp = FTP(host)
		ftp.login()
		ftp.cwd(dirname)

		
		files = []
		ftp.dir(files.append)
	
		tnow = datetime.utcnow()
		curyear = str(datetime.utcnow().year)
		recent_files = []
		for fi in files:
			fname = fi.split()[-1]
			timestamp = ' '.join(fi.split()[-4:-1])
			ftime = datetime.strptime(curyear + ' '  + timestamp, '%Y %b %d %H:%M')
			age = (tnow - ftime).total_seconds()
			if age < timelimit and fname.find('last') < 0:
				recent_files.append(fname)
		
		for fi in recent_files:
			localfile_raw = os.path.join(local_dir, fi)
			if cache and os.path.exists(localfile_raw):
				continue
			with open(localfile_raw, 'w') as f:
				print localfile_raw
				ftp.retrbinary('RETR ' + fi, f.write)

	except Exception,e:
		print e
	finally:
		ftp.close()

	print "converting files"
	print recent_files
	classpath = '/Users/ethan/insight/nexrad/toolsUI-4.3.jar'
	local_files = []
	times = []
	for fi in recent_files:
		print fi
		localfile_raw = os.path.join(local_dir, fi)
		localfile_nc = os.path.join(local_dir, fi + '.nc')

		local_files.append(localfile_nc)
		
		if not (cache and os.path.exists(localfile_nc)):
			args = ['-classpath', classpath, 'ucar.nc2.FileWriter', '-in', localfile_raw, '-out', localfile_nc]
			p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
			p.wait()

		with netCDF4.Dataset(localfile_nc, 'r') as ds:
			timestamp = ds.time_coverage_end

		ftime = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
		age = (tnow - ftime).total_seconds()
		times.append(age)

	return local_files, times

def build_prediction(station, nframes=12, interval=300, prev_frames=4):
	frames = radardb.get_latest(station, prev_frames)
	z = np.array([f['z'] for f in frames])
	frame_times = np.array([f['unix_time'] for f in frames])

	#Sort in ascending time order
	sa = np.argsort(frame_times)
	frame_times = frame_times[sa]
	z = z[sa]

	p_uniform = uniform.UniformVelocityPredictor()
	output_times = np.array([1.0*interval*i for i in range(nframes)])
	rel_times = frame_times - frame_times[-1]
	print rel_times
	prob = p_uniform.predict_prob(rel_times, z, output_times)

	pred = {}
	pred['prob'] = prob
	pred['prev_z'] = z
	pred['prev_t'] = rel_times
	pred['extent'] = frames[0]['extent']
	pred['interval'] = interval
	pred['start_time'] = frame_times[-1]

	return pred

def specific_prediction(station='kddc', nframes=20, interval=180, lon=-97.9297, lat=38.0608):
	x, y, utmzone = nexradutils.lonlat2utm(lon, lat)

	pred = build_prediction(station, nframes, interval)
	gridsize = pred['prob'][0].shape[0]
	print gridsize
	print (x,y,utmzone)
	print pred['extent']
	px, py = nexradutils.pix4coord(x, y, gridsize, extent=pred['extent'])
	print (px, py)
	trace = pred['prob'][:,py,px]
	#plt.figure()
	#plt.imshow(pred['prob'][-1], origin='lower')

	#plt.show()
	times = [interval/60. * i for i in range(nframes)]
	return times, trace, pred


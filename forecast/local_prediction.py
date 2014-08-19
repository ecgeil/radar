import netCDF4
import numpy as np
from ftplib import FTP
import subprocess
from subprocess import call
import os
from datetime import datetime
import nexradutils
import fieldshift
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

def build_prediction(station, nframes, interval):
	global debug_frames
	files, frame_times = download_latest(station, '/tmp', timelimit=1800)

	frames = []
	for fi in files:
		data = nexradutils.nexrad2utm(fi,gridsize=200)
		ftime = datetime.strptime(data['time'], '%Y-%m-%dT%H:%M:%SZ')
		z = np.nan_to_num(data['z'])
		print z.max()
		frames.append(z)

	debug_frames = frames
	prob = fieldshift.predict_prob(frames, frame_times, nframes, interval)

	pred = {}
	pred['prob'] = prob
	pred['extent'] = data['extent']
	pred['interval'] = interval
	pred['start_time'] = ftime

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
	return times, trace


'''get frames straight from server, bypassing sql cache'''

import os
import netCDF4
import nexradutils
from datetime import datetime
import StringIO
from ftplib import FTP
import subprocess
import numpy as np
import traceback
import logging
import json
import heapq
import time
import random
import string
import logging

__host = 'tgftp.nws.noaa.gov'
__ftppath = 'SL.us008001/DF.of/DC.radar/DS.p19r0/'


def get_latest(station_id, num_frames):
	logging.info("Getting data from station %s, last %d frames", station_id, num_frames)
	local_dir = "tmp/"
	unix_epoch = datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
	dirname = os.path.join(__ftppath, "SI."+station_id)
	timelimit = 3600

	ftp = None	
	frames = None
	try:
		logging.info("Connecting to host %s", __host)
		ftp = None
		ftp = FTP(__host, timeout=10) #connect to FTP server
		ftp.login()
		ftp.cwd(dirname)

		
		files = []
		ftp.dir(files.append) #get list of files
	
		tnow = datetime.utcnow()
		#server time stamps don't list the year, so we need this
		curyear = str(datetime.utcnow().year)
		recent_files = []
		recent_times = []
		ages = []
		for fi in files:
			fname = fi.split()[-1] #frame time is part of file name
			timestamp = ' '.join(fi.split()[-4:-1])
			
			ftime = datetime.strptime(curyear + ' '  + timestamp, '%Y %b %d %H:%M')
			sql_timestamp = ftime.strftime('%Y-%m-%d %H:%M:%S')
			
			unix_time = int((ftime - unix_epoch).total_seconds())
			age = (tnow - ftime).total_seconds()
			if age < timelimit and fname.find('last') < 0:
				recent_files.append(fname)
				recent_times.append(ftime)
				ages.append(age)
		
		

		#sort by age
		sa = np.argsort(ages)
		recent_files = [recent_files[i] for i in sa]
		recent_times = [recent_times[i] for i in sa]
		if len(ages) < num_frames:
			num_frames = len(ages)

		#only most recent
		ages = ages[:num_frames]
		recent_files = recent_files[:num_frames]
		recent_times = recent_times[:num_frames]

		logging.info("Grabbing %d files", len(recent_files))

		if len(recent_files) == 0:
			logger.info("The station % seems to be down--there are no recent frames.", station_id)
		
		frames = []
		for i,fi in enumerate(recent_files):
			ftime = recent_times[i]
			sql_timestamp = ftime.strftime('%Y-%m-%d %H:%M:%S')
			
			unix_time = int((ftime - unix_epoch).total_seconds())

			randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
			localfile_raw = os.path.join(local_dir, fi + randstr + ".raw")
			if False:
				continue
			with open(localfile_raw, 'w') as f: #download
				logging.debug("retrieving to " + localfile_raw)
				ftp.retrbinary('RETR ' + fi, f.write)

			
			
			localfile_nc = os.path.join(local_dir, fi + '_' + randstr + '.nc')

			#path to java conversion tool
			classpath = 'java/toolsUI-4.3.jar'
			args = ['-classpath', classpath, 'ucar.nc2.FileWriter', '-in', localfile_raw, '-out', localfile_nc]

			#convert to netcdf
			logging.debug("converting to netCDF4: " + localfile_nc)
			p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
			#wait for process to complete
			p.wait()

			#grid data
			gridsize = 200
			nx = gridsize
			ny = gridsize

			logging.debug("gridding data; grid size = %d" % gridsize)
			data = nexradutils.nexrad2utm(localfile_nc,gridsize=200)

			scaling = 1.0
			offset = 0.0

			s = StringIO.StringIO()
			np.save(s, np.nan_to_num(data['z']))
			s.seek(0)

			
			data['scaling'] = 1.0
			data['offset'] = 0.0
			data['station_id']	= station_id
			data['sql_timestamp'] = sql_timestamp
			data['unix_time'] = unix_time
			data['frame_timestamp']	= data['time'] 
			data['nx'] = nx
			data['ny'] = ny 
			data['z'] = np.nan_to_num(data['z'])

			frames.append(data)

			logging.info("Cleaning up")
			os.remove(localfile_raw)
			os.remove(localfile_nc)

	

		ftimes = np.array([f['unix_time'] for f in frames])
		sa = np.argsort(-ftimes)
		frames = [frames[i] for i in sa]
		
	except Exception,e:
		logging.exception(e)
		frames = None
	finally:
		if ftp:
			ftp.close()


	return frames

def get_latest_tz(station_id, num_frames=5):
	"""return just z and frame (unix) times"""
	frames = get_latest(station_id, num_frames)
	z = np.array([f['z'] for f in frames])
	t = np.array([f['unix_time'] for f in frames])
	return t, z
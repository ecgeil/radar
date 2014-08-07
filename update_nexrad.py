import netCDF4
import numpy as np
from ftplib import FTP
import subprocess
from subprocess import call
import os
from datetime import datetime
import time
import traceback
import nexradutils
import json
import traceback


reload(nexradutils)


host = 'tgftp.nws.noaa.gov'
dirname = 'SL.us008001/DF.of/DC.radar/DS.p19r0/SI.kbgm/'
local_dir = '/tmp'
info_fname = 'nexradinfo.json'
classpath = '/Users/ethan/dev/nexrad/toolsUI-4.3.jar'
imsize = 600

print "retrieving files"
try:
	ftp = FTP(host)
	ftp.login()
	ftp.cwd(dirname)

	files = []
	ftp.dir(files.append)
	timelimit = 3600
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
	
	recent_files_local = []
	nupdate = 0
	for fi in recent_files:
		print "checking " + fi
		localfile_raw = os.path.join(local_dir, fi)
		localfile_nc = os.path.join(local_dir, fi + '.nc')
		localfile_bin = os.path.join(local_dir, fi + '_grid.data')
		recent_files_local.append(os.path.join(local_dir,fi + '_grid.data'))
		if os.path.exists(localfile_bin):
			age = time.time() - os.path.getmtime(localfile_bin)
		if not (os.path.exists(localfile_raw) and os.path.exists(localfile_bin) and age < timelimit):
			nupdate += 1
			with open(localfile_raw, 'w') as f:
				print "fetching " + localfile_raw
				ftp.retrbinary('RETR ' + fi, f.write)
			
			
			args = ['-classpath', classpath, 'ucar.nc2.FileWriter', 
					'-in', localfile_raw, '-out', localfile_nc]
			p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
			p.wait()
			
			data = nexradutils.nexrad2utm(localfile_nc, gridsize=imsize)
			im = nexradutils.z2rgba(data['z'])
			imb = (255*im).astype('u1')
			
			with open(localfile_bin, 'w') as fraw:
				fraw.write(imb.tostring())
		
		if nupdate > 0: #update info file
			rinfo = {k:data[k] for k in ['radar_x', 'radar_y', 'station', 'vcp', 'extent']}
			rinfo['files'] = recent_files_local
			rinfo['xmin'] = data['extent'][0]
			rinfo['xmax'] = data['extent'][1]
			rinfo['ymin'] = data['extent'][2]
			rinfo['ymax'] = data['extent'][3]
			rinfo['gridsize'] = imsize
			rinfo['update_time'] = time.ctime()
			json_path = os.path.join(local_dir, info_fname)
			with open(json_path, 'w') as fjson:
				json.dump(rinfo, fjson, indent=4)
			
			
			
		
except Exception,e:
	print e
	
finally:
	ftp.close()
	print "Goodbye (python)"
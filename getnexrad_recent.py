import netCDF4
import numpy as np
from ftplib import FTP
import subprocess
from subprocess import call
import os
from datetime import datetime
import time
import traceback

host = 'tgftp.nws.noaa.gov'
ftpfname = 'sn.last'
dirname = 'SL.us008001/DF.of/DC.radar/DS.p19r0/SI.kddc'
localfile = '/tmp/nexradlast'
localfile_nc = '/tmp/nexradlast.nc'
local_dir = '/tmp'

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
	
	for fi in recent_files:
		localfile_raw = os.path.join(local_dir, fi)
		with open(localfile_raw, 'w') as f:
			print localfile_raw
			ftp.retrbinary('RETR ' + fi, f.write)

except Exception,e:
	print e
finally:
	ftp.close()
	
print "converting files"
classpath = '/Users/ethan/insight/nexrad/toolsUI-4.3.jar'
npz_files = []
for fi in recent_files:
	print fi
	localfile_raw = os.path.join(local_dir, fi)
	localfile_nc = os.path.join(local_dir, fi + '.nc')
	
	
	args = ['-classpath', classpath, 'ucar.nc2.FileWriter', '-in', localfile_raw, '-out', localfile_nc]
	p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
	p.wait()
	data = {}
	#print "opening netcdf file " + localfile_nc
	with netCDF4.Dataset(localfile_nc, 'r') as ds:
		data['refl'] = np.array(ds.variables['BaseReflectivity'][:])
		data['refl_raw'] = np.array(ds.variables['BaseReflectivity_RAW'][:])
		data['vcp'] = ds.VolumeCoveragePatternName
		data['azimuth'] = np.array(ds.variables['azimuth'][:])
		data['gate'] = np.array(ds.variables['gate'][:])
		data['latitude'] = np.array(ds.variables['latitude'][:])
		data['longitude'] = np.array(ds.variables['longitude'][:])
		data['time'] = ds.time_coverage_end
		print ds.time_coverage_end
		data['station'] = ds.ProductStation
		data['station_name'] = ds.ProductStationName
		
	
	localfile_npz = os.path.join(local_dir, 'sn-' + data['time'].replace(':','-') + '.npz')
	np.savez(localfile_npz, **data)
	npz_files.append(localfile_npz)


print 'goodbye'
	
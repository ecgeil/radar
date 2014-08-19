import netCDF4
import numpy as np
from ftplib import FTP
import subprocess
from subprocess import call
import os

host = 'tgftp.nws.noaa.gov'
ftpfname = 'sn.last'
dirname = 'SL.us008001/DF.of/DC.radar/DS.p19r0/SI.kbgm/'
localfile = '/tmp/nexradlast'
localfile_nc = '/tmp/nexradlast.nc'


print "retrieving file"
with open(localfile, 'w') as f:
	ftp = FTP(host)
	ftp.login()
	ftp.cwd(dirname)
	mtime = ftp.sendcmd('MDTM ' + ftpfname)
	ftp.retrbinary('RETR ' + ftpfname, f.write)
	#ftp.sendcmd('PASV')
	#flist = ftp.sendcmd('LIST ' + os.path.split(ftpfname)[0])
	ftp.close()
	
print "converting file"
classpath = '/Users/ethan/dev/nexrad/toolsUI-4.3.jar'
args = ['-classpath', classpath, 'ucar.nc2.FileWriter', '-in', localfile, '-out', localfile_nc]
p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
p.wait()

print "loading data"
data = {}
with netCDF4.Dataset(localfile_nc, 'r') as ds:
	data['refl'] = np.array(ds.variables['BaseReflectivity'][:])
	data['refl_raw'] = np.array(ds.variables['BaseReflectivity_RAW'][:])
	data['vcp'] = ds.VolumeCoveragePatternName
	data['azimuth'] = np.array(ds.variables['azimuth'][:])
	data['gate'] = np.array(ds.variables['gate'][:])
	data['latitude'] = np.mean(np.array(ds.variables['latitude'][:]))
	data['longitude'] = np.mean(np.array(ds.variables['longitude'][:]))
	data['time'] = ds.time_coverage_end
	data['station'] = ds.ProductStation
	data['station_name'] = ds.ProductStationName
	
	sa = np.argsort(data['azimuth'])
	data['azimuth'] = data['azimuth'][sa]
	data['refl'] = data['refl'][sa]
	data['refl_raw'] = data['refl_raw'][sa]
print 'goodbye'
	
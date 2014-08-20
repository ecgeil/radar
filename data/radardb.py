"""Functions to download, convert, and cache data from NEXRAD stations"""

import os
import pymysql
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



__dbname = 'nexrad_frames'
__host = 'tgftp.nws.noaa.gov'
__ftppath = 'SL.us008001/DF.of/DC.radar/DS.p19r0/'
__auto_purge = True

__user = 'root'
__password = 'root'

def create_frames_table():
	"""This table stores the actual frame data"""
	con = pymysql.connect(user='root', passwd=__password)
	with con:
		curs = con.cursor()

		logger.info('Creating frames table')
		curs.execute('USE %s' % __dbname)

		schema = '''CREATE TABLE IF NOT EXISTS
					frames( id INTEGER NOT NULL AUTO_INCREMENT,
							station_id CHAR(4),
							frame_datetime DATETIME,
							unix_time BIGINT,
							frame_timestamp CHAR(255),
							vcp INTEGER,
							product CHAR(255),
							zmax REAL,
							zmin REAL,
							zmean REAL,
							scaling REAL,
							offset REAL,
							clean INTEGER,
							nx INTEGER,
							ny INTEGER,
							utmzone INTEGER,
							xmin REAL,
							xmax REAL,
							ymin REAL,
							ymax REAL,
							z MEDIUMBLOB,
							PRIMARY KEY(id)
							)'''
		curs.execute(schema)
		con.commit()

		
def create_updates_table():
	con = pymysql.connect(user='root', passwd=__password)
	with con:
		curs = con.cursor()
		curs.execute('USE %s' % __dbname)

		logger.info('Creating updates table')
		schema = """CREATE TABLE IF NOT EXISTS
					updates(station_id CHAR(4) NOT NULL,
							update_time DATETIME,
							PRIMARY KEY(station_id)
						)"""
		curs.execute(schema)
		con.commit()

def needs_update(station_id, timelimit=180):
	"""Check if station needs to be updated"""
	con = pymysql.connect(user='root', passwd=__password)
	with con:
		curs = con.cursor()
		curs.execute('USE %s' % __dbname)
		curs.execute("""SELECT COUNT(1) FROM updates WHERE 
						station_id=%s AND
						update_time > ADDDATE(UTC_TIMESTAMP(), INTERVAL -%s SECOND)""",
						 [station_id, timelimit])
		res = curs.fetchone()[0]
		if res == 0:
			return True
		else:
			return False




def update_station(station_id, timelimit=3600, force_update = False):
	if not force_update:
		if not needs_update(station_id):
			logger.info("station %s is up to date", station_id)
			return

	if __auto_purge:
		purge_old()
	logger.info("Updating station %s, last %f seconds", station_id, timelimit)
	local_dir = "tmp/"
	unix_epoch = datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
	dirname = os.path.join(__ftppath, "SI."+station_id)

	con = pymysql.connect(user='root', passwd=__password)
	with con:
		curs = con.cursor()
		curs.execute('USE %s' % __dbname)

		try:
			logger.info("Connecting to host %s", __host)
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
			
			logger.info("Updating %d files", len(recent_files))
			for i,fi in enumerate(recent_files):
				ftime = recent_times[i]
				sql_timestamp = ftime.strftime('%Y-%m-%d %H:%M:%S')

				curs.execute("SELECT COUNT(1) FROM frames where frame_datetime = %s and station_id = %s", [sql_timestamp, station_id])
				nfound = curs.fetchone()[0]

				if nfound > 0:
					logger.info("skipping " + station_id + " " + sql_timestamp)
					continue
				
				unix_time = int((ftime - unix_epoch).total_seconds())

				randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
				localfile_raw = os.path.join(local_dir, fi + randstr)
				if False:
					continue
				with open(localfile_raw, 'w') as f: #download
					logger.debug("retrieving to " + localfile_raw)
					ftp.retrbinary('RETR ' + fi, f.write)

				
				
				localfile_nc = os.path.join(local_dir, fi + '_' + randstr + '.nc')

				#path to java conversion tool
				classpath = 'java/toolsUI-4.3.jar'
				args = ['-classpath', classpath, 'ucar.nc2.FileWriter', '-in', localfile_raw, '-out', localfile_nc]

				#convert to netcdf
				logger.debug("converting to netCDF4: " + localfile_nc)
				p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
				#wait for process to complete
				p.wait()

				#grid data
				gridsize = 200
				nx = gridsize
				ny = gridsize

				logger.debug("gridding data; grid size = %d" % gridsize)
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
				data['zbin'] = s.getvalue()
				

				params = [data['station_id'],
						 data['sql_timestamp'],
						 data['unix_time'],
						 data['frame_timestamp'],
						 data['vcp'],
						 data['product'],
						 float(data['zmax']),
						 float(data['zmin']),
						 float(data['zmean']),
						 float(data['scaling']),
						 float(data['offset']),
						 data['clean'],
						 float(data['nx']),
						 float(data['ny']),
						 data['utmzone'],
						 float(data['xmin']),
						 float(data['xmax']),
						 float(data['ymin']),
						 float(data['ymax']),
						 data['zbin']]
				#print len(params)
				#print "inserting %d values" % len(params)
				curs.execute( """INSERT INTO frames
									(station_id, frame_datetime,
										unix_time, frame_timestamp,
										vcp, product,
										zmax, zmin, zmean,
										scaling, offset, clean,
										nx, ny,
										utmzone,
										xmin, xmax, ymin, ymax, z)
									VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
										   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
										
							""", params)
				logger.info("Cleaning up")
				os.remove(localfile_raw)
				os.remove(localfile_nc)

			
			#modify last update time
			curs.execute( """INSERT INTO updates (station_id, update_time)
								VALUES (%s, UTC_TIMESTAMP())
								ON DUPLICATE KEY UPDATE update_time=UTC_TIMESTAMP()""",
								[station_id])

			con.commit()



		except Exception,e:
			print "got exception: "
			
			print e
			traceback.print_exc()
			logging.exception(e)
		finally:
			if ftp:
				ftp.close()

def update_stations(station_list):
	for station in station_list:
		update_station(station)


def purge_old(timelimit = 7200):
	"""drop frames older than timelimit (seconds)"""
	con = pymysql.connect(user='root', passwd=__password)
	logger.info("purging frames older than %d seconds" % timelimit)
	with con:
		curs = con.cursor()
		curs.execute('USE %s' % __dbname)
		curs.execute("""SELECT COUNT(1) from frames WHERE
						frame_datetime < ADDDATE(UTC_TIMESTAMP(), INTERVAL -%s SECOND)""",
							[timelimit])
		num_to_delete = curs.fetchone()[0]
		logger.info("purging %d frames", num_to_delete)
		
		curs.execute( """DELETE FROM frames WHERE 
						frame_datetime < ADDDATE(UTC_TIMESTAMP(), INTERVAL -%s SECOND)""", 
						[timelimit] )
		con.commit()

def get_latest(station_id, num_frames=5, update=True):
	if update:
		update_station(station_id)
	con = pymysql.connect(user='root', passwd=__password)
	logger.info("Fetching frames from %s", station_id)
	with con:
		curs = con.cursor()
		curs.execute('USE %s' % __dbname)

		curs.execute("""SELECT station_id, unix_time, vcp,
								product,
								zmin, zmax, zmean, 
								utmzone,
								xmin, xmax,
								ymin, ymax,
								z
								FROM frames WHERE
								station_id = %s
								ORDER BY frame_datetime DESC
								LIMIT %s""",
								[station_id, num_frames])

		frames = []
		for row in curs:
			data = {}
			data['station_id'] =  row[0]
			data['unix_time'] = row[1]
			data['vcp'] = row[2]
			data['product'] = row[3]
			data['zmin'] = row[4]
			data['zmax'] = row[5]
			data['zmin'] = row[6]
			data['utmzone']  = row[7]
			data['xmin'] = row[8]
			data['xmax'] = row[9]
			data['ymin'] = row[10]
			data['ymax'] = row[11]
			data['extent'] = [data['xmin'], data['xmax'], data['ymin'], data['ymax']]
			data['z'] = np.load(StringIO.StringIO(row[12]))
			frames.append(data)

		return frames

def get_latest_tz(station_id, num_frames=5, update=True):
	"""return just z and frame (unix) times"""
	frames = get_latest(station_id, num_frames, update)
	z = np.array([f['z'] for f in frames])
	t = np.array([f['unix_time'] for f in frames])
	return t, z


def continuous_update():
	with open('data/nexrad_stations.json') as f:
		stations = json.load(f)

	station_ids = stations.keys()
	station_ids = ['kdix', 'kbgm', 'kmlb']
	temp_ids = [ ]

	h = []
	for s in station_ids:
		heapq.heappush(h, (0,s))

	while True:
		t_update, station= h[0]
		if time.time() < t_update:
			time.sleep(1.0)
			continue

		logger.info("updating %s" % station)
		heapq.heappop(h)
		frames = get_latest(station, num_frames=2)
		if len(frames) == 0:
			logger.debug("got 0 frames")
			heapq.heappush(h, (time.time() + 10, station))
			continue;

		frame = frames[0]

		if frame['vcp'] in [31, 32]:
			next_update = time.time() + 1000
			logger.info("No rain, next update for %s in 1000 s" % station)
		else:
			next_update = time.time() + 180
			logger.info("Next update for %s in 180 s" % station)

		heapq.heappush(h, (next_update, station))

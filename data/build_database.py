"""Script for building the training database""''

import os,sys
import numpy as np
import nexradutils
import sqlite3
import netCDF4
import tempfile
import shutil
import datetime
import PIL
from PIL import Image


srcdir = '/Users/ethan/dev/nexrad/data/kmlb'
dstdir = '/Users/ethan/dev/nexrad/data/kmlb_grid'
sqlfile = 'kmlb.db'

tmpfpath = '/tmp/temp.nc'

product_codes = {'N0R': 'BaseReflectivity',
				'NCR': 'BaseReflectivity',
				'N0V': 'RadialVelocity',
				'NSW': 'SpectrumWidth'}

if not os.path.exists(dstdir):
	os.mkdir(dstdir)

with sqlite3.connect(os.path.join(dstdir, sqlfile)) as db:
	cursor = db.cursor()
	info_updated = False
	cursor.execute('''CREATE TABLE IF NOT EXISTS 
					radar(station TEXT,
						name TEXT,
						longitude REAL,
						latitude REAL,
						utmzone INTEGER,
						utmx REAL,
						utmy REAL)''')
		
	#cursor.execute('''DROP TABLE IF EXISTS frames''')				
	cursor.execute('''CREATE TABLE IF NOT EXISTS
					frames( fpath TEXT PRIMARY KEY,
							vcp INTEGER,
							product TEXT,
							frame_time TEXT,
							zmax REAL,
							zmin REAL,
							scaling REAL,
							offset REAL,
							clean INTEGER,
							xmin REAL,
							xmax REAL,
							ymin REAL,
							ymax REAL)
							WITHOUT ROWID''')
	
	db.commit()

	nfound = 0
	for root, dirs, files in os.walk(srcdir):
	
		for d in dirs:
			src_subdir = os.path.join(root, d)
			relpath = os.path.relpath(src_subdir, srcdir)
			dst_subdir = os.path.join(dstdir, relpath)
			if not os.path.exists(dst_subdir):
				os.mkdir(dst_subdir)
	
		for f in files:
			if f[0] == '.' or f.find('N0R') < 0:
				continue
			
				
			
			nfound += 1
			if nfound > 1e6:
				sys.exit(0)
			
			srcfile = src_subdir = os.path.join(root, f)
			relpath = os.path.relpath(src_subdir, srcdir)
			dstfile = os.path.join(dstdir, relpath)
			imfile = dstfile + '.png'
			
			cursor.execute('SELECT COUNT(1) FROM frames WHERE fpath="%s"' % imfile)
			res = cursor.fetchone()
			if res[0] > 0 and os.path.exists(imfile):
				print "skipping %s" % os.path.split(srcfile)[1]
				continue
	
			product = 'BaseReflectivity'
			for k in product_codes.iterkeys():
				if f.find(k) > 0:
					product = product_codes[k]
					product_code = k
		
			print "processing file %d: %s" % (nfound, os.path.split(srcfile)[1])
			nexradutils.raw2nc(srcfile, tmpfpath)
			#print "extracting data"
			data = nexradutils.nexrad2utm(tmpfpath, gridsize=500, product=product)
		
			z = np.clip(data['z'], 0, 75.0)
			zint = (255.0*z[::-1]/75.0).astype('u1')
			im = Image.fromarray(zint, mode='L')
			imr = im.resize((200,200), PIL.Image.ANTIALIAS)
			imr.save(imfile, "PNG")
			
			
			#image val = scaling * (data + offset)
			row = dict(vcp=data['vcp'], product=product_code, frame_time=data['time'],
						zmax=data['zmax'], zmin=data['zmin'], scaling=255.0/75.0,
						offset=0, clean=data['clean'], fpath=imfile,
						xmin=data['xmin'], xmax=data['xmax'],
						ymin=data['ymin'], ymax=data['ymax'] )
			
			cursor.execute('''INSERT OR REPLACE INTO frames VALUES(:fpath, :vcp, :product, :frame_time,
							:zmax, :zmin, :scaling, :offset, :clean, :xmin, :xmax, :ymin, :ymax)''',
							row)
		
			#use the first file to get the basic radar site info
			if not info_updated:
				cursor.execute('''DELETE FROM radar''')
				cursor.execute('''INSERT INTO radar(station, name, longitude, latitude,
													utmzone, utmx, utmy)
				 				VALUES(:station, :name, 
								:lon, :lat, :utmzone, :radar_x, :radar_y)''',
								data)
			db.commit()
			
		
		
	
			#print nfound
cursor.close()

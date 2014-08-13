import os,sys
import numpy as np
import nexradutils
import sqlite3
#import netCDF4
import tempfile
import shutil
import datetime
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import pyproj
from mpl_toolkits.basemap import Basemap
import datetime
from dateutil import parser
import predictor
from scipy.ndimage import filters, interpolation

columns = ['fpath', 'vcp', 'product', 'time', 'zmax', 'zmin', 'scaling', 'offset',
			'clean', 'xmin', 'xmax', 'ymin', 'ymax']



def gen_mask(shape, rad1=0.15, rad2=1.0):
	rad1 = rad1**2
	rad2 = rad2**2
	x = np.linspace(-1,1,shape[1])
	y = np.linspace(-1,1,shape[0])
	xx,yy= np.meshgrid(x,y)
	r2 = xx**2 + yy**2
	mask = (r2 <= rad2)*(r2 >= rad1)
	return mask

def patch_out(src):
	cy = (src.shape[0]-1)/2.0
	cx = (src.shape[1]-1)/2.0
	rmax = (min(src.shape)-1)/2.0
	patched = np.zeros(src.shape)
	for i in range(src.shape[0]):
		for j in range(src.shape[1]):
			x = j - cx
			y = i - cy
			r = (x**2 + y**2)**0.5
			if r > rmax:
				x = x*(rmax-1)/r
				y = y*(rmax-1)/r
				i2 = int(cy+y)
				j2 = int(cx+x)
			else:
				i2, j2 = i,j
			patched[i,j] = src[i2,j2]

	return patched


class FrameSource:
	def __init__(self, src_db='/Users/ethan/insight/nexrad/data/kmlb_grid/kmlb.db'):
		self.db = sqlite3.connect(src_db)
		self.curs = self.db.cursor()
		self.basemap = None

		self.curs.execute('SELECT * from radar limit 1')
		row = self.curs.fetchone()
		self.station = row[0]
		self.station_name = row[1]
		self.lon = row[2]
		self.lat = row[3]
		self.utm_zone = row[4]
		self.x = row[5]
		self.y = row[6]

		self.t0 = parser.parse('1970-01-01T00:00:00Z')

		self.curs.execute('SELECT COUNT(1) from frames')
		row = self.curs.fetchone()
		self.num_frames = row[0]

		frame = self.get_frame(0)
		self.shape = frame['z'].shape
		self.xmin, self.xmax = frame['xmin'], frame['xmax']
		self.ymin, self.ymax = frame['ymin'], frame['ymax']

		proj = pyproj.Proj(proj='utm', zone=17, ellps='WGS84')
		mco_lat, mco_lon = 28.4339,-81.325
		mco_x, mco_y = proj(mco_lon, mco_lat)
		nx, ny = self.shape

		self.pix_x = nx*(mco_x - self.xmin)/(self.xmax - self.xmin)
		self.pix_y = ny*(mco_y - self.ymin)/(self.ymax - self.ymin)
		self.markers = [[mco_x,mco_y]]


	def __del__(self):
		self.db.close()

	def row_to_frame(self, row):
		frame = {}
		for i in range(len(columns)):
			frame[columns[i]] = row[i]

		frame['time_utc'] = parser.parse(frame['time'])
		frame['time_unix'] = (frame['time_utc'] - self.t0).total_seconds()
		frame['extent'] = [frame['xmin'], frame['xmax'], frame['ymin'], frame['ymax']]
		im = Image.open(frame['fpath'])
		frame['im'] = im
		frame['z'] = np.flipud((np.array(im) - frame['offset'])/frame['scaling'])
		return frame


	def get_frame(self, n):
		self.curs.execute("SELECT * from frames limit 1 offset :offset",  dict(offset=n))
		row = self.curs.fetchone()
		#print row
		frame = self.row_to_frame(row)
		return frame

	def iter_frames(self, start=0, stop=None, step=1):
		if stop == None:
			stop = self.num_frames
		if stop < 0:
			stop = self.num_frames + stop + 1
		for i in range(start, stop, step):
			yield self.get_frame(i);



	def setup_basemap(self, frame):

		proj = pyproj.Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
		lllon, lllat = proj(frame['xmin'], frame['ymin'], inverse=True)
		urlon, urlat = proj(frame['xmax'], frame['ymax'], inverse=True)
		lon_0 = self.lon
		lat_0 = 0.0

		self.basemap = Basemap(projection='tmerc',lon_0=lon_0,lat_0=lat_0,
            k_0=0.9996,rsphere=(6378137.00,6356752.314245179),
            llcrnrlon=lllon,llcrnrlat=lllat,urcrnrlon=urlon,urcrnrlat=urlat,resolution='i')

	def draw_frame(self, frame, timestamp=True, ax=None):
		try:
			frame['time']
		except:
			frame = self.get_frame(frame)
		
		if self.basemap == None:
			self.setup_basemap(frame)

		if ax == None:
			fig = plt.figure(figsize=(4,4))
			ax = fig.add_axes([0.0,0.0,1.0,1.0])
		else:
			fig = ax.figure
		self.basemap.drawcoastlines(ax=ax)
		self.basemap.imshow(frame['z'], vmax=75, vmin=0)

		if timestamp:
			ax.text(10000, 10000, frame['time'], color='white')

		for m in self.markers:
			marker_pos = np.array(m) - np.array([self.xmin, self.ymin])
			ax.plot(marker_pos[0], marker_pos[1], 'r+', markersize=10)

		return fig

	def output_frames(self, out_dir, start, stop, step=1):
		'''Save images of frames to disk'''
		frame_no = 1
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_axes([0.0,0.0,1.0,1.0])
		for i in range(start, stop, step):
			print "frame %d" % frame_no
			fig = self.draw_frame(i, ax=ax)
			fpath = os.path.join(out_dir, "frame%d.png" % frame_no)
			frame_no += 1

			fig.savefig(fpath, transparent=True, dpi=100)
			ax.cla()

	def output_frames_constrate(self, out_dir, start, stop, frames_per_hour=10):
		'''Save images of frames to disk, maintaining an approximate rate of
		frames_per_hour'''

		frame_no = 1
		frame_idx = start
		seconds_per_frame = 3600.0/frames_per_hour
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_axes([0.0,0.0,1.0,1.0])
		frame = self.get_frame(start)
		start_time = parser.parse(frame['time'])
		while frame_idx < stop:
			ax.cla()
			#print "frame %d" % frame_no
			fig = self.draw_frame(frame, ax=ax)
			fpath = os.path.join(out_dir, "frame%d.png" % frame_no)

			fig.savefig(fpath, transparent=True, dpi=100)
			frame_no += 1
			
			target_seconds = frame_no * seconds_per_frame
			while frame_idx < stop:
				frame_idx += 1
				frame = self.get_frame(frame_idx)
				frame_time = parser.parse(frame['time'])
				delta = (frame_time - start_time).total_seconds()
				print frame_no, frame_idx, delta/(3600.0*24)
				if delta > target_seconds:
					break

	def score_predictor(self, predictor, npast=5, nfuture=12, limit=5000, prob=False):
		mask = gen_mask(self.shape, rad1=0.05, rad2=1.0)
		mask_els = np.sum(mask)
		threshold = 20
		ntotal = nfuture + npast
		
		z = np.zeros((ntotal,) + self.shape)
		times = np.zeros(ntotal)
		pred_times = []
		pred_scores = []
		zmean = []

		bin_size=300
		nbins=12
		bins = np.zeros(nbins)
		bin_counts = np.zeros(nbins)

		start_idx = 0


		for start_idx in range(0, limit, ntotal):
			
			f1 = self.get_frame(start_idx)
			
			rain_area = 1.0*np.sum(f1['z'] > threshold) / mask_els
			print start_idx, rain_area
			if rain_area < 0.03:
				print "skipping"
				continue

			now_idx = npast - 1
			for i, frame in enumerate(self.iter_frames(start_idx, start_idx+ntotal)):
				z[i] = frame['z']
				times[i] = frame['time_unix']


			times -= times[now_idx]
			
			if prob:
				pred = predictor.predict_prob(times[:now_idx+1], z[:now_idx+1], times[now_idx:])
			else:
				pred = predictor.predict(times[:now_idx+1], z[:now_idx+1], times[now_idx:])

			for i in range(nfuture):
				zidx = now_idx + i
				if prob:
					p_act = z[zidx] > threshold
					cc = np.sum((p_act[mask] - pred[i][mask])**2)/mask_els
				else:
					cc = np.corrcoef(pred[i][mask], z[zidx][mask])[0,1]
				pred_times.append(times[zidx])
				pred_scores.append(cc)
				zmean.append(rain_area)
				bin = int(times[zidx]/bin_size)
				if bin < nbins:
					bins[bin] += cc
					bin_counts[bin] += 1

		bins = bins/bin_counts

		return pred_times, pred_scores, bins


if __name__ == '__main__':
	fs = FrameSource()
	n1 = 2000
	n2 = n1+10
	z = []
	zs = []
	zp = []
	times = []
	for i in range(n1, n2):
		print i
		zi = fs.get_frame(i)['z']
		ti = fs.get_frame(i)['time_unix']
		zsi = interpolation.zoom(filters.gaussian_filter(zi,1.2),0.25)
		zpi = patch_out(zsi)
		z.append(zi)
		zs.append(zsi)
		zp.append(zpi)
		times.append(ti)

	zs = np.array(zs)
	z = np.array(z)
	times = np.array(times)

	
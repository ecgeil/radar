import numpy as np
from PIL import Image
from skimage import feature
from skimage.morphology import watershed
from skimage import measure
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse





def filter_storms(storms, min_area=4.0, min_distance=5.0):
	newstorms = []
	for i,s in enumerate(storms):
		if s.area >= min_area:
			newstorms.append(s)

	return newstorms


def find_storms(z):
	thresh_z = 20 #min z value for finding peaks
	min_distance = 5 #minimum distance between peaks
	#find peaks, using local maxima
	peaks = feature.peak_local_max(z, threshold_abs=thresh_z,
				min_distance=min_distance, indices=False)

	#uniquely label each peak
	markers, num_markers = ndimage.label(peaks)

	#use watershed algorithm to split image into basins, starting at markers
	labels = watershed(-z, markers, mask=z>thresh_z/2)

	#compute region properties
	props = measure.regionprops(labels, z)

	props = filter_storms(props)
	
	storms = []
	for p in props:
		s = {}
		s['x'], s['y'] = p.centroid
		s['centroid'] = p.centroid[::-1]
		s['max_intensity'] = p.max_intensity
		s['majlen'] = p.major_axis_length
		s['minlen'] = p.minor_axis_length
		s['angle'] = 180 - np.rad2deg(p.orientation)
		s['area'] = p.area
		storms.append(s)

	return storms


def draw_storms(storms, ax=None):
	if ax == None:
		fig = plt.figure()
		ax = fig.add_subplot(111)

	for s in storms:
		ellps = Ellipse(xy=s['centroid'], width=s['majlen'], height=s['minlen'],
					angle=s['angle'])
		ax.add_artist(ellps)
		ellps.set_clip_box(ax.bbox)
		ellps.set_alpha(0.8)
		ellps.set_facecolor('None')
		ellps.set_edgecolor('red')
		ellps.set_linewidth(3)
		plt.draw()







if __name__== '__main__':
	fpath1 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010033.png'
	#fpath1 = '/Users/ethan/insight/nexrad/testblob.png'
	fpath2 = '/Users/ethan/insight/nexrad/data/kmlb_grid/N0R/KMLB_SDUS52_N0RMLB_201306010033.png'

	im1b = np.flipud(np.array(Image.open(fpath1), dtype='uint8'))
	im2b = np.flipud(np.array(Image.open(fpath2), dtype='uint8'))

	im1 = np.flipud(np.array(Image.open(fpath1)))*75.0/255.0
	im2 = np.flipud(np.array(Image.open(fpath2)))*75.0/255.0

	thresh_z = 25
	im1t = np.where(im1 > thresh_z, im1, 0)
	im2t = np.where(im2 > thresh_z, im2, 0)

	min_distance = 8
	threshold_abs = 15
	threshold_rel = 0.05
	peaks1 = feature.peak_local_max(im1, threshold_abs=thresh_z,
				min_distance=min_distance)

	storms = find_storms(im1)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(im1, origin='lower', interpolation='nearest')
	draw_storms(storms, ax)
	plt.show()

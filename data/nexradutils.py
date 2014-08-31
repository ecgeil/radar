import numpy as np
from scipy import interpolate
import netCDF4
import pyproj
import matplotlib.pyplot as plt
import subprocess
import os
from matplotlib.mlab import griddata
import logging

def lonlat2utm(lon, lat, zone=None):
	if zone is None:
		utmzone = int(1 + (lon + 180.0)/6.0)
	else:
		utmzone = zone
	p = pyproj.Proj(proj='utm', zone=utmzone, ellps='WGS84')
	x, y = p(lon, lat)
	return x,y, utmzone

def utmbbox(lon, lat, rad, zone=None):
	'''
	Find edges of approximate UTM box centered at lon, lat which contains a circle
	of radius rad (meters)
	Returns (zone, xmin, ymin, xmax, ymax)
	where x are eastings and y are northings
	'''
	if zone == None:
		utmzone = int(1 + (lon + 180.0)/6.0)
	else:
		utmzone = zone
	g = pyproj.Geod(ellps='WGS84')
	
	azs = np.arange(0,360,1.0)
	naz = len(azs)
	lons, lats, backazs = g.fwd([lon]*naz, [lat]*naz, azs, [rad]*naz)
	
	p = pyproj.Proj(proj='utm', zone=utmzone, ellps='WGS84')
	x,y = p(lons, lats)
	return (utmzone, np.min(x), np.min(y), np.max(x), np.max(y))

def nexrad2utm(file, gridsize=100, product='BaseReflectivity', utmzone=None, width=None):
	data = {}
	with netCDF4.Dataset(file, 'r') as ds:
		sa = np.argsort(ds.variables['azimuth'])
		#print ds.variables.keys()
		refl = np.array(ds.variables[product][sa])
		az = np.array(ds.variables['azimuth'][sa])
		r = np.array(ds.variables['gate'][:])
		lat = ds.RadarLatitude
		lon = ds.RadarLongitude
		vcp = int(ds.VolumeCoveragePatternName)
		data['station'] = str(ds.ProductStation)
		data['time'] = str(ds.time_coverage_end)
		data['start_time'] = str(ds.time_coverage_start)
		data['lat'] = lat
		data['lon'] = lon
		data['name'] = ds.ProductStationName
		data['clean'] = True
	
	if len(az) < 360:
		data['clean'] = False
	
	if np.any(np.diff(r) <= 0.0):
		print "r is not monotonic"
		data['clean'] = False
	
	if np.any(np.diff(az) <= 0.0):
		data['clean'] = False
		#print "az is not monotonic"
		delindices = np.argwhere(np.diff(az) <= 0)
		az = np.delete(az, delindices)
		refl = np.delete(refl, delindices, axis=0)
	
	#nexrad distances are to the start of each gate, so center coords
	gatewidth = np.mean(np.diff(r))
	r[:-1] = 0.5*(r[1:] + r[:-1])
	r[-1] += 0.5*gatewidth
	rmax = np.max(r)
	
	#find utm bounding box around radar site
	zone, xmin, ymin, xmax, ymax = utmbbox(lon, lat, rmax*1.0, zone=utmzone)
	logging.debug("zone: %f, xmin: %f, xmax: %f, ymin: %f, ymax: %f", zone, xmin, xmax, ymin, ymax)
	
	#create a projection object to convert from lat/lon to UTM or vice-versa
	proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
	xr, yr = proj(lon, lat) #radar site UTM coordinates
	
	

	if product in ['BaseReflectivity' or 'CompositeReflectivity']:
		#refl has nans where reflectivity is below threshold.
		#Can't interpolate over nans, so set these 5 dB below min
		refl[np.isnan(refl)] = -35
	
		if vcp in [31, 32]: #clear air mode
			refl = np.clip(refl, -35, 75)
		else: #precipitation mode
			refl = np.clip(refl, 0, 75)
		#interpolate product as a function of radius and azimuth
		interp = interpolate.RectBivariateSpline(r, az, np.nan_to_num(refl.T))
	else: #needs to be fixed; only base and composite reflectivity currently work
		naz = len(az)
		zt = np.tile(refl, (2,1))
		azt = np.zeros(naz*2)
		azt[0:naz] = az - 360
		validpts = np.where(np.isfinite(refl))
		az_valid = az[validpts[0]]
		r_valid = r[validpts[1]]
		refl_valid = refl[validpts]
		az_interp = np.arange(0,360.0,1.0)
		r_interp = r[:]
		z_interp = griddata(r_valid, az_valid, refl_valid, r_interp, az_interp)
		interp = interpolate.RectBivariateSpline(r, az, np.nan_to_num(z_interp).T)

	if width is not None:
		xmin = xr - width
		xmax = xr + width
		ymin = yr - width
		ymax = yr + width
		
		
	ng = gridsize
	xcoords = np.linspace(xmin, xmax, ng, endpoint=True)
	ycoords = np.linspace(ymin, ymax, ng, endpoint=True)

	
	#define a regular grid on which to evaluate the interpolator
	xx, yy = np.meshgrid(xcoords, ycoords)
	plons, plats = proj(xx, yy, inverse=True)
	
	#geodesic calculation
	geod = pyproj.Geod(ellps='WGS84')
	
	z = np.zeros((gridsize, gridsize))
	
	rlon = np.repeat([lon], gridsize)
	rlat = np.repeat([lat], gridsize)
	for i in range(gridsize):
		azfwd, azback, dist = geod.inv(rlon, rlat, plons[i], plats[i])
		azfwd = np.mod(azfwd, 360.0)

		z[i] = np.where(dist <= rmax, interp.ev(dist, azfwd), np.nan)

	
	if product in ['BaseReflectivity' or 'CompositeReflectivity']:
		if vcp in [31, 32]: #clear air mode
			z = np.clip(z, -35, 75)
		else: #precipitation mode
			z = np.clip(z, 0, 75)
	
	xr, yr = proj(lon, lat)
	
	data['product'] = product
	data['utmzone'] = zone
	data['radar_x'] = xr
	data['radar_y'] = yr
	data['vcp'] = vcp
	data['xcoords'] = xcoords
	data['ycoords'] = ycoords
	data['extent'] = [xmin, xmax, ymin, ymax]
	data['xmin'] = xmin
	data['xmax'] = xmax
	data['ymin'] = ymin
	data['ymax'] = ymax
	data['z'] = z
	data['zmin'] = z[np.isfinite(z)].min()
	data['zmax'] = z[np.isfinite(z)].max()
	data['zmean'] = z[np.isfinite(z)].mean()
	return data
			
def pix4coord(x, y, grid_size, extent):
	"""Utility function to calculate pixel index in frame
	x, y: coordinates in the grid's units
	grid_size: number of points in each dimension of (square) grid
	extent: [xmin, xmax, ymin, ymax] in grid units
	"""
	xi = int(round(grid_size*(x - extent[0])/float(extent[1]-extent[0])))
	yi = int(round(grid_size*(y - extent[2])/float(extent[3]-extent[2])))
	xi = xi if xi < grid_size else grid_size -1
	yi = yi if yi < grid_size else grid_size -1
	xi = xi if xi >= 0 else 0
	yi = yi if yi >= 0 else 0
	return (xi, yi)



def z2rgba(z, cmap='jet', zmin=5, zmax=65):
	zscaled = (z - zmin)/(zmax-zmin)
	c = plt.get_cmap(cmap)
	im = c(zscaled)
	im[np.nan_to_num(z) < zmin] = np.array([0.0,0.0,0.0,0.0])
	im[np.isnan(z)] = np.array([0.0,0.0,0.0,0.0])
	return im

def raw2nc(infile, outfile):

	if not os.path.exists(infile):
		raise Exception("input file %s does not exist" % infile)
	classpath = '/Users/ethan/dev/nexrad/toolsUI-4.3.jar'
	args = ['-classpath', classpath, 'ucar.nc2.FileWriter', 
			'-in', infile, '-out', outfile]
	p = subprocess.Popen(['java'] + args, stdout=subprocess.PIPE)
	p.wait()
	
	
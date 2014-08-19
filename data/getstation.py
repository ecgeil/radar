import json
from pyproj import Geod


def nearest_station(lon, lat):
	with open('data/nexrad_stations.json') as f:
		stations = json.load(f)

	g = Geod(ellps='WGS84')
	min_dist = 1e10
	min_k = stations.keys()[0]
	for k in stations.keys():
		s = stations[k]
		lon_s = s['lon']
		lat_s = s['lat']
		az1, az2, dist = g.inv(lon, lat, lon_s, lat_s)
		#print k, dist/1e3
		if dist < min_dist:
			min_dist = dist
			min_k = k

	return min_k, min_dist
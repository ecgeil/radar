import requests
from pyproj import Proj

key_file_path = "data/geocoding_api_key.key"

def utmzone(lon):
	"""Find the UTM zone of a longitude
	:param lon: longitude (degrees), in [-180,180]
	:return utm zone
	"""
	return (int((lon+180)/6) % 60) + 1


def geocode(address, utm=False):
	"""
	Convert an address to coordinates (lon,lat) or (utm_zone, x, y) if utm=True
	Currently uses the Google Maps API
	:param address: some string representing a location 
					(e.g. zip code or street address)
	:param utm: if False (default), return longitude and latitude
				if True, return utm_zone, utm x coordinate, utm y coordinate
	:return coordinates, or None if the query failed
	"""
	key_file = None
	try:
		key_file = open(key_file_path, 'r')
		api_key = key_file.read()
	except Exception, e:
		raise(e)
	finally:
		if key_file != None:
			key_file.close()

	params = dict(address=address, key=api_key)
	resp = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params)
	resp = resp.json()
	if resp['status'] != 'OK':
		return None
	lat = resp['results'][0]['geometry']['location']['lat']
	lon = resp['results'][0]['geometry']['location']['lng']
	if not utm:
		return lon, lat
	else:
		zone = utmzone(lon)
		proj = Proj(proj='utm', zone=zone, ellps='WGS84')
		x,y = proj(lon,lat)
		return zone,x,y

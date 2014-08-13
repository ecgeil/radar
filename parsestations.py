import os
import re
import json

fpath = "station_list.txt"
outfile = "nexrad_stations.json"

stations = {}

with open(fpath,'r') as f:
	f.readline()
	for ln in f.readlines():
		fields = re.split(r'\W+', ln)
		#print fields
		station_id = fields[1].lower()

		latfld = 4
		while True:
			latstr = fields[latfld]
			if latstr.isdigit():
				break
			latfld += 1


		lonstr = fields[latfld+1]

		print lonstr

		lat = float(latstr[0:2]) + float(latstr[2:4])/60.0 + float(latstr[4:])/3600.
		lon = float(lonstr[0:3]) + float(lonstr[3:5])/60.0 + float(lonstr[5:])/3600.

		coords=dict(lon=-lon, lat=lat)
		stations[station_id] = coords

		print (station_id, lon, lat)


data = json.dumps(stations)

with open(outfile, 'w') as f:
	f.write(data)
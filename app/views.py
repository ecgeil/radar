from flask import render_template
from flask import request
from flask import send_from_directory
from app import app
from data import geocode
#import pymysql as mdb
from forecast import makeplot
from data import getstation
import time
import os



@app.route('/')
@app.route("/radar",)
def location_page():
	return render_template('locator.html')

@app.route("/forecast",  methods=['POST', 'GET'])
def forecast_page():
	if request.method == 'POST':
		print "got post"
		loc = request.form['location']
		coords = geocode.geocode(loc)
		if coords == None:
			return "Couldn't parse location"
		lon, lat = coords
		station, dist = getstation.nearest_station(lon, lat)
		print "station: " + station
		fname = time.strftime('%Y%m%d-%H%M%S') + ".png"
		#fname = "test.png"
		fpath = 'app/static/images/' + fname
		relpath = 'static/images/' + fname
		makeplot.makeplot(lon, lat, fpath, station)
		locdict=dict(lon=coords[0], lat=coords[1])
		return render_template('graphpage.html', loc=locdict, impath=relpath)


	return location_page()


# @app.route('/static/images/<path:filename>')
# def send_image(filename):
# 	print "here"
# 	return send_from_directory('/static/images', filename)
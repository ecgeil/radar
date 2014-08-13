from bs4 import BeautifulSoup
import requests


def getforecast(url):
	resp = requests.get(url)
	soup = BeautifulSoup(resp.text)
	time_cells = soup.find_all('span', {'class':'time'})
	forecasts = []
	for tc in time_cells:
		fc = tc.find_next().find_next()
		timestr = tc.text
		forecast = fc.text
		forecasts.append([timestr, forecast])

	return forecasts
		
if __name__=='__main__':
	
	url = "http://www.accuweather.com/en/us/orlando-international-airport-fl/32827/minute-weather-forecast/5969_poi"
	url = 'http://www.accuweather.com/en/us/athens-ga/30601/minute-weather-forecast/328217'
	forecast = getforecast(url)

import urllib
import xml.etree.ElementTree as ET

serviceurl = 'http://maps.googleapis.com/maps/api/geocode/xml?'

#address = raw_input('Enter location: ')
address = '2078 Orchard Park Dr, Niskayuna, NY 12309'
  
url = serviceurl + urllib.urlencode({'sensor':'false', 'address': address})
print 'Retrieving', url
uh = urllib.urlopen(url)
data = uh.read()
print 'Retrieved',len(data),'characters'
print data
tree = ET.fromstring(data)


results = tree.findall('result')
lat = results[0].find('geometry').find('location').find('lat').text
lng = results[0].find('geometry').find('location').find('lng').text
location = results[0].find('formatted_address').text

print 'lat',lat,'lng',lng
print location

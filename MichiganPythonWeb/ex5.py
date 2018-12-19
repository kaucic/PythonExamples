import urllib
import xml.etree.ElementTree as ET

if False:
    serviceurl = 'http://python-data.dr-chuck.net/comments_42.xml'
else:
    serviceurl = 'http://python-data.dr-chuck.net/comments_251830.xml'
 
url = serviceurl 
print 'Retrieving', url
data = urllib.urlopen(url).read()
print 'Retrieved',len(data),'characters'
#print data
tree = ET.fromstring(data)

results = tree.findall('.//count')
print results

# results is list of count xml objects
total = 0
for obj in results:
    val = obj.text
    total += int(val)

print "total=",total

#lat = results[0].find('geometry').find('location').find('lat').text
#lng = results[0].find('geometry').find('location').find('lng').text
#location = results[0].find('formatted_address').text


import json
import urllib

if False:
    url = 'http://python-data.dr-chuck.net/comments_42.json'
else:
    url = 'http://python-data.dr-chuck.net/comments_251834.json'

data = urllib.urlopen(url).read()
print 'Retrieved',len(data),'characters'

input = '''
[
  { "id" : "001",
    "x" : "2",
    "name" : "Chuck"
  } ,
  { "id" : "009",
    "x" : "7",
    "name" : "Chuck"
  } 
]'''

info = json.loads(data)
#print info

# info is a dictionary with note and comments fields
print "note=", info['note']
comments = info['comments'] # comments is a list of people dictionaries
print comments

total = 0
for item in comments:
    #print 'name=', item['name']
    #print 'count=', item['count']
    count = item['count']
    total += int(count)

print "total=",total
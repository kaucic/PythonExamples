# Note - this code must run in Python 2.x and you must download
# http://www.pythonlearn.com/code/BeautifulSoup.py
# Into the same folder as this program

import urllib
import BeautifulSoup as bs

# Start main program
if False:
    url = 'http://python-data.dr-chuck.net/comments_42.html'
else:
    url = 'http://python-data.dr-chuck.net/comments_251833.html'  

html = urllib.urlopen(url).read()
soup = bs.BeautifulSoup(html)
#print soup

# Retrieve all of the span tags
sum = 0
tags = soup('span')
for tag in tags:
    # Look at the parts of a tag
    print 'tag:',tag
    #print 'class:',tag.get('class', None)
    #print 'Contents:',tag.contents[0]
    #print 'Attrs:',tag.attrs
    sum += int(tag.contents[0])
    
print "sum=", sum

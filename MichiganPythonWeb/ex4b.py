# Note - this code must run in Python 2.x and you must download
# http://www.pythonlearn.com/code/BeautifulSoup.py
# Into the same folder as this program

import urllib
import BeautifulSoup as bs

def traverse_link(url,lid):
    html = urllib.urlopen(url).read()
    soup = bs.BeautifulSoup(html)
    #print soup
    
    # Retrieve all of the anchor tags
    tags = soup('a')
    idx = 1
    for tag in tags:
        link = tag.get('href', None)
        #print 'idx=', idx, link
        
        if idx==lid:
            break;
        idx += 1
    
    return link

# Start main program
if False:
    nurl = 'http://python-data.dr-chuck.net/known_by_Fikret.html'
else:
    nurl = 'http://python-data.dr-chuck.net/known_by_Raunuq.html'  

for i in range(7):
    nurl = traverse_link(nurl,18)
    print '\n new link', nurl

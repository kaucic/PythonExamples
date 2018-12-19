import re
import socket
import urllib

if True:
    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysock.connect(('www.pythonlearn.com', 80))
    mysock.send('GET http://www.pythonlearn.com/code/intro-short.txt HTTP/1.0\n\n')
    
    data = mysock.recv(512)
    while len(data) > 0:
        print data
        data = mysock.recv(512)
        
    mysock.close()

def find_webpage(txt):
    anchors = re.findall('\<a.*?(http:[\S]+)\"\>',txt)   
    for anchs in anchors:
        print "Found web page = ", anchs
    
    return anchors


# Start main program
#wpage = urllib.urlopen('http://www.dr-chuck.com/page1.htm')
wpage = urllib.urlopen('http://www.dr-chuck.com')
txt = wpage.read()
print txt[:20]
newpages = find_webpage(txt)
if len(newpages) > 0:
    for page in newpages:
        wpage = urllib.urlopen(page)
        txt = wpage.read()
        print txt[:20]
        
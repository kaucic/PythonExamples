import sqlite3
import re

conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('''DROP TABLE IF EXISTS Counts;''')
cur.execute('''CREATE TABLE Counts (org TEXT, count INTEGER);''')

fname = 'mbox.txt'
fh = open(fname)
for line in fh:
    if line.startswith('From: '):
        pieces = line.split()
        email = pieces[1]
        print "email address = ", email
        sub_pieces = email.split('@')
        org = sub_pieces[1]
        #org = re.search('\@([0-9|a-z|A-Z|\.]+)',email)
        print "email org =", org
        cur.execute('SELECT count FROM Counts WHERE org = ? ;', (org, ))
        row = cur.fetchone()
        if row is None:
            cur.execute('''INSERT INTO Counts (org, count) VALUES ( ?, 1 );''', (org, ))
        else : 
            cur.execute('UPDATE Counts SET count=count+1 WHERE org = ? ;', (org, ))
    
conn.commit()
fh.close()

# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

print
print "Counts:"
for row in cur.execute(sqlstr) :
    print str(row[0]), row[1]

cur.close()


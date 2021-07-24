#import sqlalchemy
#from sqlalchemy import create_engine
#engine = create_engine('mysql+',echo=False);
#conn = engine.connect()
#cur = engine.execute("SELECT * FROM Rating;")

import sqlite3
conn = sqlite3.connect('.\output\database.sqlite')

cur = conn.cursor()    
cur.execute('SELECT * FROM Persons')   
data = cur.fetchall()
    
print()

for row in data:
   print(row)

print()

cur.close()
conn.close()

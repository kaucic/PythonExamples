#import sqlalchemy
#from sqlalchemy import create_engine
#engine = create_engine('mysql+://root:Cath0212@localhost:3306/test',echo=False);
#conn = engine.connect()
#cur = engine.execute("SELECT * FROM Rating;")

import pymysql
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='Cath0212', db='test')
cur = conn.cursor()
cur.execute("SELECT * FROM Rating;")
print(cur.description)

print()

for row in cur:
   print(row)

cur.close()
conn.close()

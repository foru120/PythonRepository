from sqlite3 import *
conn = connect('C:/sqlite/Score.db')
csr = conn.cursor()
csr.execute("SELECT * FROM Score ORDER BY score DESC")
result = csr.fetchall()
for r in result:
    print(r)

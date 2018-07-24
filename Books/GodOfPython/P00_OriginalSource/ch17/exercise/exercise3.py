from sqlite3 import *
from random import randint
conn = connect("C:/sqlite/Score.db")
csr = conn.cursor()

while True:
    print("이름과 국가를 입력하시오.")
    name = input("이름:")
    nation = input("국가:")
    nation = nation.lower().capitalize()
    csr.execute("SELECT * from Score WHERE name==?",(name,))
    if csr.fetchall():
        print("이미 해당 이름이 있습니다. 재입력 해주세요.")
    else:
        csr.execute("""INSERT INTO Score(name, last_stage, score, nation)
                    VALUES(?,?,?,?)""", (name, randint(1,10), randint(1,99999), nation.capitalize()))
        csr.close()
        conn.commit()
        conn.close()
        break

import sqlite3
import random

conn = sqlite3.connect('Score.db')
c = conn.cursor()

while True:
    name = input('name >')
    nation = input('nation >')
    last_stage = random.randint(1,11)
    score = random.randint(0, 100000)

    c.execute('select * from score where name = ?', (name,))

    if c.fetchone():
        print('이미 존재하는 사용자 입니다. 다시 입력해주세요.')
    else:
        c.execute('insert into score values(?,?,?,?)', (name, last_stage, score, nation.capitalize()))
        print('입력이 완료되었습니다.')
        break

c.close()
conn.commit()
conn.close()
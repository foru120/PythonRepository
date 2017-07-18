# word2.py
import sqlite3

conn = sqlite3.connect('C:/gop/ch17/mydb.db')   # 해당 경로의 mydb.db를 읽는다.
                                                # 없다면 새로 생성

c = conn.cursor()

c.execute('''INSERT INTO mytable(word, meaning, level)
VALUES("python",
"A python is a large snake that kills animals by squeezing them
with its body.",
2)''')

c.execute('''INSERT INTO mytable(word, meaning, level)
VALUES("sql",
"structured query language: a computer programming language used
for database management.",
1)''')

c.execute('''INSERT INTO mytable(word, meaning, level)
VALUES("apple",
"An apple is a round fruit with smooth green, yellow, or red skin
and firm white flesh.",
1)''')

c.execute('''SELECT * from mytable''')
print(c.fetchall())  # 테이블의 마지막 레코드까지 출력

c.close()
conn.commit()
conn.close()

# word3.py
import sqlite3

conn = sqlite3.connect("mydb.db")
c = conn.cursor()

word = input("word : ")  # 단어 입력
meaning = input("meaning : ")  # 단어 뜻 입력
level = input("level : ")  # 단어 레벨 입력
c.execute("INSERT INTO mytable VALUES(?,?,?)", (word, meaning, level))  # 변수 전달

c.execute("SELECT * FROM mytable")
print(c.fetchall())

c.close()
conn.commit()
conn.close()

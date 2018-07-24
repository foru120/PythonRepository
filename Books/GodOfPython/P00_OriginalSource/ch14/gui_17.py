# gui_17.py
import tkinter  # python 3.x
from tkinter import filedialog  # python 3.x
# import Tkinter as tkinter #python 2.x
# import tkFileDialog as filedialog #python 2.x

root = tkinter.Tk()
frame = tkinter.Frame(root, bg='green')
frame.pack()


def openfile():
    f = filedialog.askopenfile()  # 파일 객체 반환
    filepath.set(f.name)  # 파일 객체의 name 속성은 해당 파일의 경로 문자열이다.
    f.close()  # 함수가 종료되기 전에 파일을 닫는다.

filepath = tkinter.StringVar()
filepath.set("filepath")

button = tkinter.Button(frame, text='open', command=openfile)
button.grid(row=0, column=0)

label_path = tkinter.Label(frame, textvariable=filepath)
label_path.grid(row=1, column=0)

root.mainloop()

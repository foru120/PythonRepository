# gui_17_1.py
import tkinter  # python 3.x
from tkinter import filedialog  # python 3.x
# import Tkinter as tkinter          #python 2.x
# import tkFileDialog as filedialog  #python 2.x

root = tkinter.Tk()
frame = tkinter.Frame(root, bg='green')
frame.pack()

f = None  # f는 openfile함수에서 열게될 파일객체를 참조할 변수


def openfile():
    global f
    f = filedialog.askopenfile("rb")
    filepath.set(f.name)

filepath = tkinter.StringVar()
filepath.set("filepath")


def saveasfile():
    global f
    a = f.read()  # 기존 파일(f)의 내용을 읽는다.
    if a is None:  # 빈 파일이라면
        return  # 함수 종료
    else:
        save_f = filedialog.asksaveasfile("wb")
        # 새로운 파일을 만들어 파일 객체를 반환
        filepath.set(save_f.name)
        save_f.write(a)  # 새 파일에 기존 파일의 내용 기록
    f.close()  # 기존 파일(f)을 닫는다.
    save_f.close()  # 새로 만든 파일을 닫는다.


button_open = tkinter.Button(frame, text='open', command=openfile)
button_open.grid(row=0, column=0)
button_save_af = tkinter.Button(frame, text='save as file', command=saveasfile)
button_save_af.grid(row=0, column=1)
label_path = tkinter.Label(frame, textvariable=filepath)
label_path.grid(row=1, column=0)
root.mainloop()

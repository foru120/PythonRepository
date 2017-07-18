# gui_20.py
import tkinter
from tkinter import filedialog

root = tkinter.Tk()


def openfile():
    global f
    f = filedialog.askopenfile(mode='r+')
    root.title(f.name)
    contents = f.readlines()
    text.delete("1.0", 'end')
    text.insert("1.0", ''.join(contents))


def savefile():
    global f
    if not f:
        f = filedialog.asksaveasfile()
    contents = text.get("1.0", "end")
    f.seek(0, 0)
    f.write(contents)
    f.flush()


def saveasfile():
    global f
    f.close()

    save_f = filedialog.asksaveasfile()
    root.title(save_f.name)
    f = save_f
    contents = text.get("1.0", "end")
    save_f.write(contents)
    f.flush()


def exit():
    f.close()
    raise SystemExit

f = None

# 메뉴만들기
menubar = tkinter.Menu(root)
root['menu'] = menubar

menu_file = tkinter.Menu(menubar, tearoff=0)
menubar.add_cascade(menu=menu_file, label="File")
menu_file.add_command(label="Open", command=openfile)
menu_file.add_command(label="Save", command=savefile)
menu_file.add_command(label="Save as", command=saveasfile)
menu_file.add_command(label="Exit", command=exit)

# 텍스트 부품 설정

text = tkinter.Text(root, width=40, height=10)  # width는 문자수 height는 줄 수
text.pack()

root.mainloop()

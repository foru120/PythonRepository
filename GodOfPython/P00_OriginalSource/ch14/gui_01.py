# gui_01.py
import tkinter

root = tkinter.Tk()
frame = tkinter.Frame(root, height=100, width=100, relief='sunken',
                      bd=2, bg='#D9E5FF')
frame.pack()  # 배치관리자 pack 호출
root.mainloop()  # root와 그에 포함된 자식부품들의 상태를 계속해서 갱신한다.

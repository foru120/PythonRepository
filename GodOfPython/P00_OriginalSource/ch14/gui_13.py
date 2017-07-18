# gui_13.py
import tkinter

root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.pack()

check_A = tkinter.IntVar()  # 제어변수 check_A
check_B = tkinter.StringVar()  # 제어변수 check_B

cb_A = tkinter.Checkbutton(frame, text='check Test1', variable=check_A,
                           onvalue=1, offvalue=0)
cb_A.grid(row=0, column=0)
label_A = tkinter.Label(frame, textvariable=check_A)
label_A.grid(row=1, column=0)

cb_B = tkinter.Checkbutton(frame, text='check Test2', variable=check_B,
                           onvalue="on", offvalue="off")
cb_B.grid(row=0, column=1)
label_B = tkinter.Label(frame, textvariable=check_B)
label_B.grid(row=1, column=1)
root.mainloop()

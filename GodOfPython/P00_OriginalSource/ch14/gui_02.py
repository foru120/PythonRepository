# gui_02.py
import tkinter


def change_bg():
    btn.configure(background='green')

root = tkinter.Tk()
root.geometry("{}x{}".format(100, 100))
frame_up = tkinter.Frame(root, height=60, width=90, background='blue')
frame_down = tkinter.Frame(root, height=30, width=90)
frame_up.pack()
frame_down.pack()

btn = tkinter.Button(frame_down, text='click', command=change_bg,
                     foreground='white', background='black',
                     activeforeground='blue',
                     activebackground='#FF007F')
btn.pack()

root.mainloop()

# gui_11.py
import tkinter


def callback():
    root.title("Hello Python")

root = tkinter.Tk()
frame = tkinter.Frame(root, padx=100, pady=50)
frame.pack()
button = tkinter.Button(frame, text='click', command=callback)
button.pack()
root.mainloop()

# gui_11_1.py
import tkinter


def callback():
    root.title("Hello Python")

root = tkinter.Tk()
frame = tkinter.Frame(root, padx=100, pady=50)
frame.pack()

button = tkinter.Button(frame, text='click')
button.pack()

label = tkinter.Label(frame, text='click')
label.pack()

button.bind("<ButtonPress-1>", lambda e: callback())
button.bind("<Double-1>", lambda e: root.title("Mouse Double click"))
button.bind("<ButtonPress-3>", lambda e: root.title("Mouse Right click"))
label.bind("<Double-2>", lambda e: root.title("tkinter Label event"))

root.mainloop()

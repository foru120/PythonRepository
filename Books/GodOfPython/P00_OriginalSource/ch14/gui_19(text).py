# gui_19.py
import tkinter

root = tkinter.Tk()
text = tkinter.Text(root, width=35, height=15)
text.pack()

text.insert('1.0', 'python is simple')
text.insert('1.10', 'very ')
text.insert('1.21', '!')

root.mainloop()

# gui_06.py
import tkinter

root = tkinter.Tk()
cvs = tkinter.Canvas(root, width=220, height=50)
cvs.pack()
builtin_bitmap = ['error', 'gray12', 'gray25', 'gray50', 'gray75',
                  'hourglass', 'info', 'questhead', 'question', 'warning']
for i in range(0, len(builtin_bitmap)):
    cvs.create_bitmap(20 * i + 20, 30, bitmap=builtin_bitmap[i])

root.mainloop()

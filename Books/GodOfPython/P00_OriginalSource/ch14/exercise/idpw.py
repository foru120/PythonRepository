import tkinter

def reset():
    text_id.set('')
    text_pw.set('')

def login():
	f = open ("C:/gop/ch14/exercise/idpw.txt", 'r')
	try:
		for s in f:
			idpw = s.split('/')
			if (idpw[0] == text_id.get()) and (idpw[1].strip()==text_pw.get()):
				frame.destroy()
				label = tkinter.Label(text = "{} login..".format(text_id.get()))
				label.pack()
				return 0
		reset()
	except:
		print("exception occured")
	finally:
		f.close()


root = tkinter.Tk()
root.title("login")
root.geometry('180x120')
text_id = tkinter.StringVar(value='')
text_pw = tkinter.StringVar(value='')
frame = tkinter.Frame(root)
frame.pack()
button = tkinter.Button(frame, text = 'reset', command = reset)
button.grid(row=0, column=0, columnspan = 2)

button = tkinter.Button(frame, text = 'login', command = login)
button.grid(row=3, column=0, columnspan = 2)

label = tkinter.Label(frame, text = 'ID')
label.grid(row = 1, column = 0)
entry_id = tkinter.Entry(frame, textvariable = text_id)
entry_id.grid(row = 1, column = 1)
label = tkinter.Label(frame, text = 'PW')
label.grid(row = 2, column = 0)
entry_pw = tkinter.Entry(frame, textvariable = text_pw, show='*')
entry_pw.grid(row = 2, column = 1)
root.mainloop()

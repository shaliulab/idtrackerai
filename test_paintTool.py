import Tkinter, tkSimpleDialog
root = Tkinter.Tk() # dialog needs a root window, or will create an "ugly" one for you
root.withdraw() # hide the root window
password = tkSimpleDialog.askstring("Password", "Enter password:", show='*', parent=root)
root.destroy() # clean up after yourself!

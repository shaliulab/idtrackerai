from Tkinter import *
import tkMessageBox

window = Tk()
window.wm_withdraw()

#message at x:200,y:200
window.geometry("1x1+200+200")#remember its .geometry("WidthxHeight(+or-)X(+or-)Y")
tkMessageBox.showerror(title="error",message="Error Message",parent=window)

#centre screen message
window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))
tkMessageBox.showinfo(title="Greetings", message="Hello World!")

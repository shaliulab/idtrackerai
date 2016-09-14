from Tkinter import *
import sys

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Navigate video")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.b=Button(master,text="Insert next segment to inspect",command=self.popup)
        self.b.pack()
        # self.b2=Button(master,text="print value",command=lambda: sys.stdout.write(self.entryValue()+'\n'))
        # self.b2.pack()

    def popup(self):
        self.w=popupWindow(self.master)
        self.master.wait_window(self.w.top)

    def entryValue(self):
        return self.w.value


if __name__ == "__main__":
    root=Tk()
    m=mainWindow(root)
    root.mainloop()

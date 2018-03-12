from tkinter import *


class Application(Frame):
    def send(self, *args, **kwargs):
        # saved_args = locals()
        # print("saved_args is", saved_args)
        if self.ENTRY.get() != "":
            self.TEXT.config(state=NORMAL)
            # Paste user entry
            user_input = "Vous : " + self.ENTRY.get() + "\r\n"
            self.TEXT.insert(END, user_input)
            print(self.ENTRY.get())

            # Now paste chatbot response
            bot_response = "Solutec : " + "" + "\r\n"
            self.TEXT.insert(END, bot_response)

            # Update some settings
            self.TEXT.config(state=DISABLED)
            self.TEXT.see("end")
            self.ENTRY.delete(0, "end")
        else:
            print("Empty entry")

    def createWidgets(self):
        self.TEXT = Text(self, bg="#bbd1f9", font="Arial 12 bold")
        self.TEXT.insert(END, "Solutec : Comment puis-je vous etre utile ?\r\n")
        self.TEXT.config(state=DISABLED)
        self.TEXT.grid(row=0, column=0, columnspan=2, sticky=N+S+E+W)

        # Scrollbar
        # self.SCROLL = Scrollbar(self, command=self.TEXT)

        self.ENTRY = Entry(self, bg="#eaf7ec", font="Arial 12 bold", highlightcolor="#80aef7", width=70,
                           highlightthickness="1", relief="groove")
        self.ENTRY.grid(row=1, column=0, sticky=W)
        self.ENTRY.focus()

        self.SEND = Button(self, text="Send", font="System", fg="White", bg="#1dd138", width="15",
                           command=self.send)
        self.SEND.grid(row=1, column=1, sticky=E)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
        self.bind_all("<Return>", self.send)


root = Tk()
root.title("Chatbot SOLUTEC")
#oot.geometry("500x500")
app = Application(master=root)
app.mainloop()
root.destroy()


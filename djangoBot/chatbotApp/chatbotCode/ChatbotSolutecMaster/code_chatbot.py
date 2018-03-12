# this is the executable

def test555(entry):
    return "Hello"


### Installation commands
import pip
pip.main(["install","tensorflow"])
pip.main(["install","tflearn"])
pip.main(["install","nltk"])
pip.main(["install","django"])
# importation des librarie
import nltk
nltk.download("punkt")


#on met le stem en francais
stemmer = nltk.stem.SnowballStemmer('french')

# things we need for Tensorflow
import numpy as np
import tflearn

#importation de tensorflow
import tensorflow as tf
import random

# importation de la bibliotheque et du fichier json pour l'ia
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']
# On parcours les phrases
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # on divise la phrase en mots
        w = nltk.word_tokenize(pattern)
        # on ajoute les mots a la liste
        words.extend(w)
        # on joute les mots au ducument avec le tag correspondant
        documents.append((w, intent['tag']))
        # ajoute les tags a la liste de classe
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# on trie et on normalise les mots on enleve les doublons
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# On enlever les doublons
#trie dans l'ordre alphabetique
classes = sorted(list(set(classes)))

#affichages des differentes classe
print (len(documents), "documents", documents)
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


print("Initialisation de l'entrainement")
# Création de l entrainement
training = []
output = []

# création d'un tableau vide
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # on initilise toute les sortie du reseau neuronne a 0 et on passe a 1 celle qui est attendu
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    #on passe la valeur a l'entrainement
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

def clean_up_sentence(sentence):
    # permet de decouper la phrase en mot
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("trouver dans la categorie: %s" % w)

    return(np.array(bag))

p = bow("Ou puis-je renconter Solutec ?", words)
mdl_pred = model.predict([p])
print (p)
print("class")
print (classes)
print("model prediction")
print(mdl_pred)

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
#trouve la bonne phrases selon le odele etablit par l'entrainement
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

#Choix de la reponse
def response(sentence, userID, show_details):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return (random.choice(i['responses']))
            results.pop(0)


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
            bot_response = "Solutec : " + response(self.ENTRY.get(), '123', False) + "\r\n"
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


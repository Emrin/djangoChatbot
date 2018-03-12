# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:44:28 2017

@author: adela
"""

# importation des librairies 
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

#on met le stem en francais
stemmer = nltk.stem.SnowballStemmer('french')

# importation du fichier json pour l'ia dans un string
with open('intents.json') as json_data:
    intents = json.load(json_data)
    
#definition des differentes classes et sous classes
words = []
classes = []
documents = []
ignore_words = ['?']

# On parcours les phrases qui etaiant contenu dans le json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # on divise la phrase en mots
        w = nltk.word_tokenize(pattern)
        # on ajoute les mots a la liste
        words.extend(w)
        # on joute les mots au ducument avec le tag correspondant
        documents.append((w, intent['tag']))
        #
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Cette étapes consiste a normaliser les mots pour enlever le conjugaison, la grammaire, ...
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# On enlever les doublons
classes = sorted(list(set(classes)))

print(words)
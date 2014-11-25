#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      alex.sternberg
#
# Created:     23/11/2014
# Copyright:   (c) nykos 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from sklearn.naive_bayes import MultinomialNB
import os
from os.path import isfile, join
import re
from email.parser import Parser
import nltk
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from Tkinter import *



messages_dir_train = """C:\\TRAINING\\"""
messages_dir_test = """C:\\TESTING\\"""
labels_filename = """c:\\SPAMTrain.label"""

class Classifier():
    def __init__(self):
        self.clf = LinearSVC()
        self.vect = TfidfVectorizer()
        self.extract = None

    def train(self, messages, labels):
        text = []
        for m in messages:
            s = BeautifulSoup(m.as_string())
            text.append(s.get_text())
        features = self.vect.fit_transform(text)
        self.clf.fit(features,np.array(labels))

    def teach(self, message, label):
        pass

    def generate_features(self, message):
        return self.vect.transform([BeautifulSoup(message.as_string()).get_text()])

    def get_class(self, message):
        return self.clf.predict(self.generate_features(message))

class Application(Frame):
    def predict(self):
        print "hit predict"
        p = Parser()
        message = p.parse(self.T.get())
        if classifier.get_class(message)[0] == '0':
            print "Spam"
        else: print "Ham"

    def createWidgets(self):
        self.S = Scrollbar(self)
        self.T = Text(self, height=50, width=50)
        self.S.pack(side=RIGHT, fill=Y)
        self.T.pack(side=LEFT, fill=Y)
        self.S.config(command=self.T.yview)
        self.T.config(yscrollcommand=self.S.set)
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit

        self.QUIT.pack({"side": "bottom"})

        self.predict = Button(self)
        self.predict["text"] = "Predict",
        self.predict["command"] = self.predict

        self.predict.pack({"side": "bottom"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

classifier = Classifier()

def main():
    fignum = 1
    labels = []
    training_messages = []
    testing_messages = []
    print "getting messages and labels"
    for line in open(labels_filename):
        m = re.match('(?P<label>[01]{1}) (?P<filename>.+\.eml)', line)
        if (m):
            #print "got match: " + m.group('filename') + "(" + m.group('label') + ")"
            labels.append(m.group('label'))
            p = Parser()
            message = p.parse(open(join(messages_dir_train,m.group('filename'))))
            if isinstance(message, list):
                for m in message:
                    training_messages.append(m)
                    labels.append(m.group('label'))
            else:
                training_messages.append(message)

    #print training_messages

    classifier.train(training_messages, labels)

    for file in os.listdir(messages_dir_test):
        message = p.parse(open(join(messages_dir_test,file)))
        if isinstance(message, list):
            for m in message:
                testing_messages.append(m)
        else:
            testing_messages.append(message)

    print classifier.get_class(testing_messages[10])

    root = Tk()
    app = Application(master=root)
    app.mainloop()
    root.destroy()

if __name__ == '__main__':
    main()








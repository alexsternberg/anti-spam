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
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from Tkinter import *
import ttk



messages_dir_train = """C:\\TRAINING\\"""
messages_dir_test = """C:\\TESTING\\"""
labels_filename = """c:\\SPAMTrain.label"""

labels = []
filenames = []
training_messages = []
testing_messages = []


def load_messages(name):
    p = Parser()
    message = p.parse(open(join(messages_dir_train,name)))
    training_messages.append(message)

class Classifier():
    def __init__(self):
        self.clf = LinearSVC()
        self.vect = TfidfVectorizer()
        self.extract = None

    def train(self, features):
        self.clf.fit(features,np.array(labels))

    def teach(self, message, label):
        pass

    def generate_features(self, message):
        return self.vect.transform([BeautifulSoup(message.as_string()).get_text()])

    def get_class(self, message):
        return self.clf.predict(self.generate_features(message))

class Application(Frame):
    def predict(self):
        #print "hit predict"
        p = Parser()
        message = p.parsestr(self.T.get("1.0",END))
        if self.cls.get_class(message)[0] == '0':
            print "Spam"
            self.TEXT.configure(text="SPAM", fg="white", bg="red")
            app.update()
        else:
            print "Ham"
            self.TEXT.configure(text="HAM", fg="white", bg="green")
            app.update()

    def train(self):
        print "loading messages"
        for name in filenames:
            load_messages(name)
            self.PG.step(1)
            app.update()
        self.cls = Classifier()
        print"parsing messages"
        text = []
        for m in training_messages:
            s = BeautifulSoup(m.as_string())
            text.append(s.get_text())
            self.PG.step(1)
            app.update()
        print "building vectorizer"
        features = self.cls.vect.fit_transform(text)
        self.PG.step(500)
        app.update()
        print "training machine"
        self.cls.train(features)
        self.PG.step(500)
        app.update()
        print "done!"

    def createWidgets(self):
        self.S = Scrollbar(self)
        self.T = Text(self, height=20, width=50)
        self.S.pack(side=RIGHT, fill=Y)
        self.T.pack(side=TOP, fill=Y)
        self.S.config(command=self.T.yview)
        self.T.config(yscrollcommand=self.S.set)

        self.TEXT = Label(self, text="WAIT", bg="grey", fg = "black")
        self.TEXT.pack(side=LEFT)

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack(side=RIGHT)

        self.TRAIN = Button(self)
        self.TRAIN["text"] = "Train",
        self.TRAIN["command"] = self.train

        self.TRAIN.pack(side=RIGHT)

        self.PREDICT = Button(self)
        self.PREDICT["text"] = "Predict",
        self.PREDICT["command"] = self.predict

        self.PREDICT.pack(side=LEFT)



        self.PG = ttk.Progressbar(orient=HORIZONTAL, length=450, mode='determinate', maximum=2*len(labels)+1000)
        self.PG.pack(side=BOTTOM)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

classifier = Classifier()

def main():
    fignum = 1
    print "getting messages and labels"
    for line in open(labels_filename):
        m = re.match('(?P<label>[01]{1}) (?P<filename>.+\.eml)', line)
        if (m):
            #print "got match: " + m.group('filename') + "(" + m.group('label') + ")"
            labels.append(m.group('label'))
            filenames.append(m.group('filename'))

    #print training_messages

    #classifier.train(training_messages, labels)



if __name__ == '__main__':
    main()

def p():
    p = Parser()
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

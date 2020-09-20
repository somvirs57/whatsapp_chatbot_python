# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:35:55 2020

@author: somvi
"""

import numpy as np
import pickle 
import random
import json

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Chatbot:
    
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
    
    def __init__(self,intents, model, tokenizer, max_length):
        self.intents = intents
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length 
        
    def start_chat(self):
        user_response = input("Say something\n")
        while not self.make_exit(user_response):
            user_response = input(self.generate_response(user_response))
            
    def make_exit(self, user_response):
        for word in self.exit_commands:
            if word == user_response:
                print("Have a nice day!")
                return True
            else:
                return False
        
        
    def generate_response(self,user_response):
        stemmed_sentence = self.get_stemmed_sentence(user_response)
        tag = self.get_predicted_tag_class(stemmed_sentence)
        reverse_dict = self.get_reverse_dict()
        tag_name = reverse_dict[tag]
        
        for intent in self.intents['intents']:
            if intent['tag'] == tag_name:
                response = random.choice(intent['responses'])
                
        return response
        
    def get_predicted_tag_class(self,stemmed_sentence):
        seq = tokenizer.texts_to_sequences([stemmed_sentence])
        pad_seq = pad_sequences(seq,maxlen=self.max_length,padding='post')
        x = model.predict(pad_seq)
        y = np.argmax(x)
        
        return y
    
    def get_reverse_dict(self):
        tags = []
        for intent in self.intents['intents']:
            tags.append(intent['tag'])
        reverse_dict = dict((key,value) for key,value in enumerate(tags))
        
        return reverse_dict
                
    def get_stemmed_sentence(self,sentence):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        sent = ' '.join(stemmer.stem(token) for token in nltk.word_tokenize(sentence) 
                        if token not in stop_words)
        return sent

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to the whatsapp chatbot for businesses"

@app.route("/bot", methods=["GET","POST"])
def reply_whatsapp():

    msg = request.form.get('Body')  
    
    chat = Chatbot(intents, model, tokenizer, max_length)
    bot_response = chat.generate_response(msg)

    response = MessagingResponse()
    response.message(bot_response)
    return str(response)



if __name__ == "__main__":
    
    with open ('intents.json', 'r') as f:
        intents = json.load(f)
    
    model = load_model("ret_chatbot.h5")

    with open('tokenizer.pickle', 'rb') as infile:
        tokenizer = pickle.load(infile)
        
    with open('max_seq_length', 'rb') as infile:
        max_length = pickle.load(infile)
        
    #chat = Chatbot(intents, model, tokenizer, max_length)
    #chat.start_chat()
    
    app.run()
        
        
        
        
        
        
        
        
        
        
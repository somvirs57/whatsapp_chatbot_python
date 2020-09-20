# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:04:16 2020

@author: somvi
"""

import json
import pickle

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')

import pandas as pd

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input,Dense,Embedding,LSTM,Dropout
from tensorflow.keras.models import Model


class Data_processing:
    
    def __init__(self, intents):
        self.intents = intents
        self.embed_dim = 32
        self.lstm_dim = 64
        
    def get_tag_response(self):
        tags = []
        xy = []
        for intent in self.intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for sentence in intent['patterns']:
                xy.append((sentence,tag))  
        return tags, xy
    
    def get_dataframe(self):
        tags,xy = self.get_tag_response()
        df = pd.DataFrame(xy, columns=['sentence', 'tag'])
        tag_map = dict()
        for index,tag in enumerate(tags):
            tag_map[tag] = index
        df['tag'] = df['tag'].map(tag_map)
        df = df.sample(frac=1)
        return df
    
    def get_stemmed_dataframe(self):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        df = self.get_dataframe()
        df['stemmed_sentence'] = df['sentence'].apply(lambda sentence: 
                                                      ' '.join(stemmer.stem(token) 
                                                      for token in nltk.word_tokenize(sentence) if token not in stop_words))
        return df
        
    def get_training_xy(self):
        tags,_ = self.get_tag_response()
        df = self.get_stemmed_dataframe()
        tags_cat = df['tag']
        labels = to_categorical(tags_cat, len(tags))
        
        inputs = df['stemmed_sentence']
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(inputs)
        sequences = tokenizer.texts_to_sequences(inputs)
        max_length = max(len(x) for x in sequences)
        paded_seq = pad_sequences(sequences,maxlen=max_length,padding='post')
        vocab = len(tokenizer.word_index)+ 1
        
        with open('tokenizer.pickle','wb') as outfile:
            pickle.dump(tokenizer,outfile)
        print('Tokenizer saved successfully')
        
        with open('max_seq_length','wb') as outfile:
            pickle.dump(max_length,outfile)

        return paded_seq,labels,max_length, vocab  
       
    def train_data(self,embed_dim=32,lstm_dim=64):
        print(f'{embed_dim}\n{lstm_dim}')
        paded_seq,labels, max_length,vocab = self.get_training_xy()
        i = Input(shape=(max_length,))
        x = Embedding(vocab,self.embed_dim)(i)
        x = Dropout(0.5)(x)
        x = LSTM(self.lstm_dim)(x)
        x = Dense(120,activation='relu')(x)
        x = Dense(5,activation='softmax')(x)
        
        model = Model(i,x)

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
       
        r = model.fit(paded_seq,labels,epochs=100)
        
        model_name = "ret_chatbot.h5"
        model.save(model_name)
        print(f"{model_name} saved successfully")
        
        return r
        
if __name__ == "__main__":
    
    with open ('intents.json', 'r') as f:
        intents = json.load(f)
        
    data = Data_processing(intents)
    history = data.train_data(embed_dim=42,lstm_dim=120)
    
    
    
    
    
    
    
    
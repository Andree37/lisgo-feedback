#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:27:32 2020

@author: andre
"""

import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

from stop_words import stop_words

# Initial run - uncomment
#import nltk
#nltk.download('punkt')
#nltk.download('vader_lexicon')

def get_text(text_file):
    text = []
    with open(text_file, 'r') as file:
        for line in file:
            clear_line = line.replace('\n', '').replace(',', '').strip()
            text.append(clear_line)
                
    str_text = " ".join(text)
    return str_text


# reading text
def clean_text(preprocessed_text):
    # converting to lowercase
    lower_case = preprocessed_text.lower()
    
    # Removing punctuations
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    
    # splitting text into words
    tokenized_words = cleaned_text.split()

    # Removing stop words from the tokenized words list
    final_words = [word for word in tokenized_words if word not in stop_words]
    
    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)
        
    return lemma_words
    

# Get emotions text
def get_emotions(emotion_file, text):
    emotion_list = []
    final_words = clean_text(text)
    with open(emotion_file, 'r') as file:
        for line in file:
            clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in final_words:
                emotion_list.append(emotion)
                
    return emotion_list


def plot_emotions(emotion_list):
    w = Counter(emotion_list)
    print(w)

    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    fig.set_size_inches(12, 5)
    plt.savefig('graph.png')
    plt.show()
    
    
    
    
    
    
    

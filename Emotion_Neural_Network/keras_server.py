#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:18:03 2020

@author: andre
"""
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pickle
from collections import OrderedDict
import operator

from flask import Flask, request
app = Flask(__name__)

model = load_model("best_model.hdf5")
MAX_SEQUENCE_LENGTH=20
THRESHOLD = 0.1

emotions_dic = {
    0: 'anger', 
    1: 'boredom',
    2: 'empty', 
    3: 'enthusiasm',
    4: 'fun',
    5: 'happiness', 
    6: 'hate',
    7: 'love', 
    8: 'neutral',
    9: 'relief', 
    10: 'sadness', 
    11: 'surprise',
    12: 'worry'
}
    
@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json()['text']
    
    # get tokenizer from model creation
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        
    seq= loaded_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    
    # probs
    probs = pred.tolist()
    
    classes = np.argmax(pred, axis = 1)

    result = classes.tolist()[0]
    emotion = emotions_dic.get(result)
    
    all_emotion_certainty = {}
    for index in emotions_dic.keys():
       all_emotion_certainty[emotions_dic.get(index)] = probs[0][index]
       
    od = OrderedDict(sorted(all_emotion_certainty.items(), key=operator.itemgetter(1), reverse=True))
    
    threshold_dic = {}
    for k,v in od.items():
        if v > THRESHOLD:
            threshold_dic[k] = v
    
    json_str = json.dumps([{'emotion': emotion}, {'certainty': probs[0][result]},
                           {'emo_certain_threshold': threshold_dic}])

    return json_str



if __name__ == '__main__':
    app.run(debug=True, port=5051)
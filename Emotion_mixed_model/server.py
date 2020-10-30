#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 04:27:30 2020

@author: andre
"""

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import json
import pickle
import itertools
from collections import Counter
from flask import Flask, request
from flask_pymongo import PyMongo
# change dir or __init__ for packaging
import sys
sys.path.insert(1, '/Users/andre/sensor-sentiment-analysis/Emotion_Detection_wordmatch/')
from util_functions import get_emotions
app = Flask(__name__)
app.config['MONGO_DBNAME'] = "feedbacklisgo"
app.config["MONGO_URI"] = "mongodb://admin:admin@cluster0-shard-00-00.mhwna.mongodb.net:27017,cluster0-shard-00-01.mhwna.mongodb.net:27017,cluster0-shard-00-02.mhwna.mongodb.net:27017/lisgofeedback?ssl=true&replicaSet=atlas-8afz0c-shard-0&authSource=admin&retryWrites=true&w=majority"
mongo = PyMongo(app)

keras_model = load_model("../Emotion_Neural_Network/best_model.hdf5")
MAX_SEQUENCE_LENGTH=20
THRESHOLD = 0.1
CERTATINTY_WEIGHT = 0.4

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

@app.route('/')
def home():
    html = """
    <h1>Hello, please write a text and see if the emotion json returned matches what you wrote, thank you!</h1>
    <h2>Please use this website carefuly, its purpose is to test the model only :D</h2>
    <form action="/predict" method="post">
  <input size="100" type="text" name="text"></input>
  <input type="submit" value="Emotion"></input>
</form> """
    return html


    
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    k_main_emotion, k_certainty, k_all_emotions = keras_predict(text)
    w_pred = word_matching_predict(text)
    dict_pred = dict(w_pred)
    
    # # add some certainty to the keras model pred
    sum_values = 0
    for v in w_pred.values():
        sum_values += v 
    
    od = {k: v for k, v in sorted(dict_pred.items(), key=lambda item: item[1], reverse=True)}
    for k,v in od.items():
        emotion = k.strip()
        certainty = k_all_emotions.get(emotion)
        weighted_certainty = certainty + (certainty * v /sum_values * CERTATINTY_WEIGHT)
        k_all_emotions[emotion] = weighted_certainty
        
    ordered_k_all_emotions = {k: v for k, v in sorted(k_all_emotions.items(), key=lambda item: item[1], reverse=True)}
    return detected_emotions(ordered_k_all_emotions)



@app.route('/neg_feedback/<emotions_dict>', methods=['POST'])
def neg_feedback(emotions_dict):
    feedback = request.form.get('feedback')
    mongo.db.feedback.insert_one({'emotions': emotions_dict, 'feedback': 'negative',
                                  'correction_emotion': feedback})
    home = """<h1>Thank you for your time :D</h1>
    <button onclick="document.location.href='/';">Click To Give more Feedback!</button>
    """
    return home



@app.route('/user_feedback/<write_dict>', methods=['POST'])
def user_feedback(write_dict):
    response = request.form.get('feedback')
    if response == "yes":
        return agree(write_dict)
    else:
        return disagree(write_dict)
    
    

def agree(emotions_dict):
    mongo.db.feedback.insert_one({'emotions': emotions_dict, 'feedback': 'positive'})
    home = """<h1>Thank you for your time :D</h1>
    <button onclick="document.location.href='/';">Click To Give more Feedback!</button>
    """
    return home
    


def disagree(emotions_dict):
    html = f"""
    <h1>What should the main emotion be?</h1>
    <form action='/neg_feedback/{emotions_dict}' method="post">
    <input type="radio" id="anger" name="feedback" value="anger">
    <label for="anger">Anger</label><br>
    <input type="radio" id="boredom" name="feedback" value="boredom">
    <label for="boredom">Boredom</label><br>
    <input type="radio" id="empty" name="feedback" value="empty">
    <label for="empty">Empty</label><br>
    <input type="radio" id="enthusiasm" name="feedback" value="enthusiasm">
    <label for="enthusiasm">Enthusiasm</label><br>
    <input type="radio" id="fun" name="feedback" value="fun">
    <label for="fun">Fun</label><br>
    <input type="radio" id="happiness" name="feedback" value="happiness">
    <label for="happiness">Happiness</label><br>
    <input type="radio" id="hate" name="feedback" value="hate">
    <label for="hate">Hate</label><br>
    <input type="radio" id="love" name="feedback" value="love">
    <label for="love">Love</label><br>
    <input type="radio" id="neutral" name="feedback" value="neutral">
    <label for="neutral">Neutral</label><br>
    <input type="radio" id="relief" name="feedback" value="relief">
    <label for="relief">Relief</label><br>
    <input type="radio" id="sadness" name="feedback" value="sadness">
    <label for="sadness">Sadness</label><br>
    <input type="radio" id="surprise" name="feedback" value="surprise">
    <label for="surprise">Surprise</label><br>
    <input type="radio" id="worry" name="feedback" value="worry">
    <label for="worry">Worry</label><br>
    <input type="submit" value="Send"></input>
</form>
    """
    return html
    


def detected_emotions(emotions_dict):
    top5_emotions = itertools.islice(emotions_dict.items(), 0, 5)
    html = """
    <h1>Top 5 Detected Emotions:</h1>
    """
    for k,v in top5_emotions:
        html += f'<h2>{k}:</h2> <p>{v} accuracy</p>'
        
    write_dict = json.dumps(emotions_dict)
        
    html += "<h2>Agree?</h2>"
    html += f"""<form action='/user_feedback/{write_dict}' method="post">
  <input type="radio" id="yes" name="feedback" value="yes">
  <label for="yes">Yes</label><br>
  <input type="radio" id="no" name="feedback" value="no">
  <label for="no">No</label><br>
  <input type="submit" value="Send"></input>
  </form> """
    return html
    

    
def word_matching_predict(text):
    emotions_file = "../Emotion_Detection_wordmatch/13emotions.txt"
    emotions = get_emotions(emotions_file, text)

    counter = Counter(emotions)
    
    return counter


    
def keras_predict(text):
    # get tokenizer from model creation
    with open('../Emotion_Neural_Network/tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        
    seq= loaded_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = keras_model.predict(padded)
    
    # probs
    probs = pred.tolist()
    
    classes = np.argmax(pred, axis = 1)

    result = classes.tolist()[0]
    emotion = emotions_dic.get(result)
    
    all_emotion_certainty = {}
    for index in emotions_dic.keys():
       all_emotion_certainty[emotions_dic.get(index)] = probs[0][index]
    
    certainty = probs[0][result]
    
    return emotion, certainty, all_emotion_certainty



if __name__ == '__main__':
    app.run(debug=True, port=5052)
    
    

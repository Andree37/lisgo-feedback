#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:18:03 2020

@author: andre
"""
from collections import Counter

from flask import Flask, request
from util_functions import get_emotions
app = Flask(__name__)

@app.route('/')
def home():
    html = """
    <h1>Hello, please write a text and see if the emotion json returned matches what you wrote, thank you!</h1>
    <form action="/predict" method="post">
  <input size="100" type="text" name="text"></input>
  <input type="submit" value="Emotion"></input>
</form> """
  
    return html
    
  


@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json()['text']
    
    emotions_file = "13emotions.txt"
    emotions = get_emotions(emotions_file, text)

    w = Counter(emotions)
    
    return {"counter": w}


if __name__ == '__main__':
    app.run(debug=True, port=5053)

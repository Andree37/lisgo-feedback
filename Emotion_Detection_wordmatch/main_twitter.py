#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:26:22 2020

@author: andre
"""

from util_functions import get_text, get_emotions, plot_emotions

# Running the file
if __name__ == "__main__":
    tweets = get_text('tweets-nl.txt')
    emotions_file = "13emotions.txt"
    emotions = get_emotions(emotions_file, tweets)
    
    plot_emotions(emotions)
    











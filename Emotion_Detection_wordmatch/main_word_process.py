#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:25:20 2020

@author: andre
"""

from util_functions import get_text, get_emotions, plot_emotions

# Running the file
if __name__ == "__main__":
    read_text = get_text('read.txt')
    emotions_file = "emotions.txt"
    emotions = get_emotions(emotions_file, read_text)
    
    plot_emotions(emotions)
    
    
    
    
    
    
    
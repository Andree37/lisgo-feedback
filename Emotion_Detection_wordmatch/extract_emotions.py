#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:58:48 2020

@author: andre
"""
# Get emotion dic
emotion_pairs = {}
with open('emotions.txt', 'r') as file:
    for line in file:
            clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            emotion_pairs[word] = emotion
            
# Get unique emotions
unique_emotions = []
for v in emotion_pairs.values():
    if v not in unique_emotions:
        unique_emotions.append(v)
    

'''[' sad', ' cheated', ' singled out', ' loved', ' attracted', ' fearful', 
' happy', ' angry', ' apathetic', ' esteemed', ' anxious', ' lustful', 
' attached', ' free', ' embarrassed', ' powerless', ' surprise', ' fearless', 
' bored', ' safe', ' adequate', ' belittled', ' hated', ' independent', 
' codependent', ' average', ' obsessed', ' entitled', ' alone', ' focused', 
' demoralized', ' derailed', ' ecstatic', ' lost', ' burdened']
'''

# Translate these emotions to the 12 main emotions from emotions_dic
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

translation_dic = {}
translation_dic['sad'] = 'sadness'
translation_dic['cheated'] = 'sadness'
translation_dic['singled out'] = 'sadness'
translation_dic['loved'] = 'love'
translation_dic['attracted'] = 'love'
translation_dic['fearful'] = 'worry'
translation_dic['happy'] = 'happiness'
translation_dic['angry'] = 'anger'
translation_dic['apathetic'] = 'empty'
translation_dic['esteemed'] = 'enthusiasm'
translation_dic['anxious'] = 'worry'
translation_dic['lustful'] = 'enthusiasm'
translation_dic['attached'] = 'worry'
translation_dic['free'] = 'fun'
translation_dic['embarrassed'] = 'anger'
translation_dic['powerless'] = 'empty'
translation_dic['surprise'] = 'surprise'
translation_dic['fearless'] = 'enthusiasm'
translation_dic['bored'] = 'boredom'
translation_dic['safe'] = 'neutral'
translation_dic['adequate'] = 'relief'
translation_dic['belittled'] = 'sadness'
translation_dic['hated'] = 'hate'
translation_dic['independent'] = 'enthusiasm'
translation_dic['codependent'] = 'empty'
translation_dic['average'] = 'boredom'
translation_dic['obsessed'] = 'worry'
translation_dic['entitled'] = 'enthusiasm'
translation_dic['alone'] = 'boredom'
translation_dic['focused'] = 'neutral'
translation_dic['demoralized'] = 'empty'
translation_dic['derailed'] = 'sadness'
translation_dic['ecstatic'] = 'enthusiasm'
translation_dic['lost'] = 'empty'
translation_dic['burdened'] = 'worry'

translated_emotions = {}
for k,v in emotion_pairs.items():
    translated_emotions[k] = translation_dic.get(v.strip())
    
print(translated_emotions)

# Write to the new file the translated emotions
with open('13emotions.txt', 'w') as file:
    for k,v in translated_emotions.items():
        file.write(f"'{k}': '{v}'\n")
    file.close()


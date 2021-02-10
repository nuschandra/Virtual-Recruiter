##prepare dataset
import nltk
from nltk.tag import hmm
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import json
import string

with open('train_Introductions.json') as file:
    jdata = json.load(file)

train_list = []
for data in jdata['Introductions']:
    sentlist=[]
    sentlist.append(('<S>','O'))
    for sequence in data['data']:
        if 'entity' not in sequence:
            tokenList = sequence['text'].lower().split()
            for tok in tokenList:
                sentlist.append((tok,'O'))
        else:
            tokenList = sequence['text'].lower().split()
            for idx,tok in enumerate(tokenList):
                if idx:
                    sentlist.append((tok,'I_'+sequence['entity']))
                else:
                    sentlist.append((tok,'B_'+sequence['entity']))  
    sentlist.append(('<\S>','O'))
    train_list.append(sentlist)
    
print (len(train_list))
print(train_list)

# Import HMM module
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_list)
print (tagger)

test = "I am from the vpdp team"
print (tagger.tag(test.split()))

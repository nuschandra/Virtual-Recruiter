import pandas as pd
import json
import string

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

with open("intents.json") as file:
    jdata=json.load(file)
    sentences=[]
    labels=[]
    for intent in jdata["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern.lower())
            labels.append(intent["tag"])
    
    df = pd.DataFrame({'text':sentences,
                       'intent':labels})
    df.to_csv('train.csv',index=False)

    print(df)

with open("test_intents.json") as file:
    jdata=json.load(file)
    sentences=[]
    labels=[]
    for intent in jdata["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern.lower())
            labels.append(intent["tag"])
    
    df = pd.DataFrame({'text':sentences,
                       'intent':labels})
    df.to_csv('test.csv',index=False)

    print(df)

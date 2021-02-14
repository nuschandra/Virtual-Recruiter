import random
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import os
import numpy as np
import pandas as pd
import json

train = pd.read_csv("train.csv")
model=keras.models.load_model('bert_intent_detection.hdf5',custom_objects={"BertModelLayer": BertModelLayer})
tokenizer = FullTokenizer(vocab_file="vocab.txt")
classes = train.intent.unique().tolist()
print(classes)
with open('intents.json') as file:
    data = json.load(file)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        test_sentence=[]
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        sentences=[inp]
        pred_tokens = map(tokenizer.tokenize, sentences)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids +[0]*(21-len(tids)),pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))
        print(pred_token_ids)
        predictions = model.predict(pred_token_ids).argmax(axis=-1)
        final_intent=''
        for text, label in zip(sentences, predictions):
            final_intent=classes[label]
            print("text:", text, "\nintent:", classes[label])
            print()
        
        for tg in data["intents"]:
            if tg['tag'] == final_intent:
                responses = tg['responses']
        print(random.choice(responses))
chat()
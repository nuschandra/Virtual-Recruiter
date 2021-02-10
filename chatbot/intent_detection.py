import nltk
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import json
import string

with open('intents.json') as file:
    data = json.load(file)


words = []
labels = []
docs_x = []
docs_y = []

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern.lower().translate(remove_punct_dict))
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

with open('train_Introductions.json') as file:
    jdata = json.load(file)

for dat in jdata['Introductions']:
    line=""
    for sequence in dat['data']:
        line += sequence['text']
    
    wrds = nltk.word_tokenize(line.lower().translate(remove_punct_dict))
    words.extend(wrds)
    docs_x.append(wrds)
    docs_y.append("introductions")

    if "introductions" not in labels:
        labels.append("introductions")

words = LemTokens(words)
words = sorted(list(set(words)))

labels = sorted(labels)

print(words)
print(labels)

training = []
output = []

# Creating BAG OF WORDS (BOW) features for the training sentences
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [lemmer.lemmatize(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


# Training of NN model with the BoW features
training = numpy.array(training)
output = numpy.array(output)

model = keras.Sequential()
model.add(layers.Dense(8,input_shape=(len(training[0]),)))
model.add(layers.Dense(8))
model.add(layers.Dense(len(output[0]), activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(training, output, epochs=1000, batch_size=8)

model.save('intent_detection')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s.lower().translate(remove_punct_dict))
    s_words = [lemmer.lemmatize(word) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        model=keras.models.load_model('intent_detection')
        test=bag_of_words(inp, words).reshape(-1,len(training[0]))
        results = model.predict(test)
        print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
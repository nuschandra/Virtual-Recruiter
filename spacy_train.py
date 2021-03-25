import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import json
import os
import warnings

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

def create_data(data_path):
    training_data = []
    with open(data_path) as data:
        lines = data.readlines()
    for line in lines:
        resume = json.loads(line)
        text = resume['content']
        annotations = resume['annotation']
        labels = []
        for annotation in annotations:
            points = annotation['points'][0]
            start = points['start']
            end = points['end'] + 1
            if len(annotation['label'])>0:
                if annotation['label'][0]=='Name':
                    print(points['text'])
            else:
                print(text)
            labels.append((start,end,annotation['label'][0]))
        training_data.append((text,{"entities":labels}))
    return training_data

def train_blank_spacy_model(training_data):
    for text, annot in tqdm(training_data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                #print(doc)
                print(start)
                print(end)
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents # label the text with the ents
        db.add(doc)

    db.to_disk("train.spacy") # save the docbin object

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   training_data = create_data("ResumeTrainingData.json")
   train_blank_spacy_model(training_data)

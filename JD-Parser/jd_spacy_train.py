import json
import os
import spacy
import random
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
from spacy.util import minibatch,compounding
from spacy._ml import create_default_optimizer
from thinc.neural import Model
import logging

def create_data(data_path):
    try:
        training_data = []
        lines=[]
        with open(data_path, 'r',encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            data = json.loads(line)
            text = data['content']
            entities = []
            if data['annotation'] is not None:
                for annotation in data['annotation']:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]
    
                    for label in labels:
                        #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                        entities.append((point['start'], point['end'] + 1 ,label))
    
    
                training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + data_path + "\n" + "error = " + str(e))
        return None

train_data_exp = pickle.load(open('ner_experience_data.pkl','rb'))

nlp_exp_train_model = spacy.blank('en')

n_iter =20

def train_model_exp(train_data_exp):
    if 'ner' not in nlp_exp_train_model.pipe_names:
        ner= nlp_exp_train_model.create_pipe('ner')
        nlp_exp_train_model.add_pipe(ner, last=True)
        
    for _, ann in train_data_exp:
        for ent in ann['entities']:
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp_exp_train_model.pipe_names if pipe != 'ner']
    with nlp_exp_train_model.disable_pipes(*other_pipes):
        optimizer = nlp_exp_train_model.begin_training()
        for itn in range(n_iter):
            print('Starting iter : '+str(itn))
            random.shuffle(train_data_exp)
            losses = {}
            
            for text, an in train_data_exp:
                try:
                    if len(text)>0:
                        nlp_exp_train_model.update([text], [an], drop=0.2, sgd=optimizer, losses=losses)
                except Exception as e:
                    print(e)
                    pass
            print("Losses", losses)
        
train_model_exp(train_data_exp)

nlp_exp_train_model.to_disk('nlp_exp_model')
        

def train_blank_spacy_model(training_data,validation_data,test_data):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner =  nlp.create_pipe("ner")
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe("ner")
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    warnings.filterwarnings("once", category=UserWarning, module='spacy')
    dummy_optimizer=create_default_optimizer(Model.ops)
    dummy_optimizer.learn_rate=0
    dummy_optimizer.alpha=0
    dummy_optimizer.b1=0
    dummy_optimizer.b2=0
    model_number=0
    loss_value=[]
    val_loss_value=[]
    optimizer = nlp.begin_training()
    batch_sizes=compounding(1.0,4.0,1.5)
    for iteration in range(100):
        print(iteration)
        random.shuffle(training_data)
        batches=minibatch(training_data,size=batch_sizes)
        losses={}
        #for resume_text, annotation in training_data:
        #    nlp.update([resume_text], [annotation], drop = 0.2, sgd=optimizer, losses=losses)
        for batch in batches:
            text, annotations = zip(*batch)
            try:
                nlp.update(text, annotations, drop = 0.2, sgd=optimizer, losses=losses)
            except Exception as e:
                    pass
        print("Losses ({}/{})".format(iteration+1,100), losses)
        loss_value.append(losses)
        
        validation_loss={}
        for val_text,val_annotations in validation_data:
            try:
                nlp.update([val_text],[val_annotations],sgd=dummy_optimizer,losses=validation_loss)
            except Exception as e:
                    pass
        
        print("Validation Losses ({}/{})".format(iteration+1,100), validation_loss)
        val_loss_value.append(validation_loss)
        if(iteration%10==0):
            nlp.to_disk('jd_model'+str(model_number))
            model_number+=1
    print(loss_value)
    print(val_loss_value)
    nlp.to_disk('jd_model')
    return loss_value,val_loss_value



training_data = create_data("jd_anno_train.json")
validation_data=create_data("jd_anno_test.json")
test_data=create_data("jd_anno_test.json")
loss_value,val_loss_value = train_blank_spacy_model(training_data,validation_data,test_data)

import matplotlib.pyplot as plt
plt.plot(range(0,100),[i['ner'] for i in loss_value],  label='Training Data ')
plt.plot(range(0,100),[i['ner'] for i in val_loss_value],   label='Testing Data')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss Values')


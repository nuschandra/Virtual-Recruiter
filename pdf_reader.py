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

    
    optimizer = nlp.begin_training()
    for iteration in range(100):
        random.shuffle(training_data)
        losses={}
        for resume_text, annotation in training_data:
            nlp.update([resume_text], [annotation], drop = 0.2, sgd=optimizer, losses=losses)
        print("Losses = ", losses)
    
    nlp.to_disk('resume_model')

    examples = create_data("ResumeTestingData.json")
    tp=0
    tr=0
    tf=0

    ta=0
    c=0        
    for text,annot in examples:

        f=open("resume"+str(c)+".txt","w")
        doc_to_test=nlp(text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)

        for i in set(d.keys()):

            f.write("\n\n")
            f.write(i +":"+"\n")
            for j in set(d[i]):
                f.write(j.replace('\n','')+"\n")
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[0,0,0,0,0,0]
        for ent in doc_to_test.ents:
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
            if(d[ent.label_][0]==0):
                #f.write("For Entity "+ent.label_+"\n")   
                #f.write(classification_report(y_true, y_pred)+"\n")
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))
    


#trained_ner_model = train_blank_spacy_model(training_data)
#trained_ner_model.to_disk("resume_model")
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   training_data = create_data("ResumeTrainingData.json")
   train_blank_spacy_model(training_data)

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
    for iteration in range(200):
        print(iteration)
        random.shuffle(training_data)
        batches=minibatch(training_data,size=batch_sizes)
        losses={}
        #for resume_text, annotation in training_data:
        #    nlp.update([resume_text], [annotation], drop = 0.2, sgd=optimizer, losses=losses)
        for batch in batches:
            text, annotations = zip(*batch)
            nlp.update(text, annotations, drop = 0.2, sgd=optimizer, losses=losses)
 
        print("Losses ({}/{})".format(iteration+1,200), losses)
        loss_value.append(losses)
        
        validation_loss={}
        for val_text,val_annotations in validation_data:
            nlp.update([val_text],[val_annotations],sgd=dummy_optimizer,losses=validation_loss)
        
        print("Validation Losses ({}/{})".format(iteration+1,200), validation_loss)
        val_loss_value.append(validation_loss)


        if(iteration%25==0):
            nlp.to_disk('resume_model'+str(model_number))
            model_number+=1

    print(loss_value)
    print(val_loss_value)
    nlp.to_disk('resume_model')

    nlp = spacy.load('resume_model3')

    examples = create_data("ResumeTestingData.json")
    tp=0
    tr=0
    tf=0

    ta=0
    c=0  
    measurement={}
    measurement['Skills']=[0,0,0,0,0]
    measurement['Experience']=[0,0,0,0,0]
    measurement['Degree']=[0,0,0,0,0]      
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
                if(ent.label_=='Skills'):
                    (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,pos_label='Skills',average='binary')
                    a=accuracy_score(y_true,y_pred)
                    d[ent.label_][0]=1
                    d[ent.label_][1]+=p
                    d[ent.label_][2]+=r
                    d[ent.label_][3]+=f
                    d[ent.label_][4]+=a
                    d[ent.label_][5]+=1

                    measurement[ent.label_][0]+=p
                    measurement[ent.label_][1]+=r
                    measurement[ent.label_][2]+=f
                    measurement[ent.label_][3]+=a
                    measurement[ent.label_][4]+=1
                if(ent.label_=='Degree'):
                    (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,pos_label='Degree',average='binary')
                    a=accuracy_score(y_true,y_pred)
                    d[ent.label_][0]=1
                    d[ent.label_][1]+=p
                    d[ent.label_][2]+=r
                    d[ent.label_][3]+=f
                    d[ent.label_][4]+=a
                    d[ent.label_][5]+=1

                    measurement[ent.label_][0]+=p
                    measurement[ent.label_][1]+=r
                    measurement[ent.label_][2]+=f
                    measurement[ent.label_][3]+=a
                    measurement[ent.label_][4]+=1

                if(ent.label_=='Experience'):
                    (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,pos_label='Experience',average='binary')
                    a=accuracy_score(y_true,y_pred)
                    d[ent.label_][0]=1
                    d[ent.label_][1]+=p
                    d[ent.label_][2]+=r
                    d[ent.label_][3]+=f
                    d[ent.label_][4]+=a
                    d[ent.label_][5]+=1

                    measurement[ent.label_][0]+=p
                    measurement[ent.label_][1]+=r
                    measurement[ent.label_][2]+=f
                    measurement[ent.label_][3]+=a
                    measurement[ent.label_][4]+=1

        c+=1
    for i in measurement:
        print(measurement)
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((measurement[i][3]/measurement[i][4])*100)+"%")
        print("Precision : "+str(measurement[i][0]/measurement[i][4]))
        print("Recall : "+str(measurement[i][1]/measurement[i][4]))
        print("F-score : "+str(measurement[i][2]/measurement[i][4]))
        


#trained_ner_model = train_blank_spacy_model(training_data)
#trained_ner_model.to_disk("resume_model")
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   training_data = create_data("ResumeTrainingData.json")
   validation_data=create_data("ResumeValidationSet.json")
   test_data=create_data("ResumeTestingData.json")
   train_blank_spacy_model(training_data,validation_data,test_data)

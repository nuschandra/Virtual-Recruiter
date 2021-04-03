# -*- coding: utf-8 -*-
"""
Created on Thu Feb 4 19:02:56 2021

@author: Lakshmi Subramanian
"""
import pandas as pd
import nltk
import re
import spacy
import pickle
import random
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from sklearn.metrics import classification_report
from tensorflow.keras import backend as k


def degree_extraction(sentences):
    cleaned_str = []
    for sentence in sentences.split("."):
        if (re.search("degree|education|university", sentence, flags=re.IGNORECASE)):
            cleaned_str.append(sentence)
    return (".".join(cleaned_str))

def clean_text(x):
    return  x.replace("\r\n"," ").replace(";",".").replace("-","").lstrip(".").strip()

grammar_exp = r"""
    NP1: {<JJR>?<IN>?<JJ|JJS|NNP>?<CD><NN|NNS><IN>?<JJ>?<NN|RB>+<IN|JJ><DT>?<JJ>?<NN|NNP>+} 
    NP2: {<NNP>?<NN><IN>+?<JJS>?<CD><NN|NNS>}
    NP3: {<IN>?<NNP|JJ|JJS>+<CD><NN|NNS>+<VBG|JJ>?<NN><IN><NN|NNP>?<.+>+} 
    NP4: {<IN>?<NNP|JJ|JJS>+<CD><NN|NNS><RB|JJ>+<VBG>?<IN>?<NN|NNP><.+>+} 
    NP5: {<JJR>?<IN>?<JJ|JJS|NNP>?<CD><NN|NNS><IN>?<JJ>?<NN|RB|VBG>+<IN|JJ><DT>?<JJ>?<.+>+}
    NP6: {<CD><NN|NNS><RB|JJ>+<VBG>?<IN>?<NN|NNP>+}
    NP7: {<IN>?<NNP|JJ|JJS>+<CD><NN|NNS>+<IN>?<VBG|JJ>+<NN>} 
    """
grammar_skill = r"""
    NP1: {<DT>?<JJ>*<NN|NNS|NNP|NNPS>+}
    NP2: {<IN>?<JJ|NN>*<NNS|NN>}
    VS: {<VBG|VBZ|VBP|VBD|VB|VBN><NNS|NN>*}
    Commas: {<NN|NNS>*<,><NN|NNS>*<,><NN|NNS>*} 
"""

def get_experience(x):
    x = x.replace("\r\n"," ").replace(";",".").replace("-","").replace(':',".").split('.')
    for i in range(len(x)):
        if ' year ' in x[i].lower() or ' yrs' in x[i].lower() or ' years ' in x[i].lower():
            a= x[i].strip()
            tokens = word_tokenize(str(a))
            tokens_pos = pos_tag(tokens)
            #print(tokens_pos)

            cp = nltk.RegexpParser(grammar_exp)
           # print(a)
            noun_phrases = []
            for np_chunk in cp.parse(tokens_pos):
                if isinstance(np_chunk, nltk.tree.Tree) and np_chunk.label() in ( 'NP1','NP2','NP3','NP4','NP5','NP6','NP7'):
                    # if np_chunk is of grammer 'NP' then create a space seperated string of all leaves under the 'NP' tree
                    noun_phrase = ""
                    for (org, tag) in np_chunk.leaves():
                        noun_phrase += org + ' '

                    noun_phrases.append(noun_phrase.rstrip())
            if len(noun_phrases) >0 :
                #print(noun_phrases)
                return(noun_phrases)

            
mystopwords=stopwords.words("English") + ['experience','computer,','science','expert','knowledge','plus','proficiency','understanding','excellent','ability','skill','responsibility']
WNlemma = nltk.WordNetLemmatizer()

df = pd.read_csv('data job posts.csv')
df = df[df['IT']==True].copy()
df = df[~df.Title.str.contains("Ad ")].copy()
df = df.reset_index(drop=True)
df = df[df['RequiredQual'].notnull()].copy()
df = df.reset_index(drop=True)

df['exp'] = df['jobpost'].apply(get_experience)
df['deg'] = df['RequiredQual'].apply(lambda x : degree_extraction(x.replace(';','.').replace('-','').lstrip()))


df1 = df.copy()
df1 = df1[df1.deg != ''].copy()
df1["jobpost"]=df1["jobpost"].apply(clean_text)
df1["deg"]=df1["deg"].apply(clean_text)
df1["RequiredQual"]=df1["RequiredQual"].apply(clean_text)
df1["Title"]=df1["Title"].apply(clean_text)


list_ner = []
for i in range(len(df1['jobpost'])):
    tup_degree = (df1.iloc[i]['jobpost'].find(df1.iloc[i]['deg']), int(df1.iloc[i]['jobpost'].find(df1.iloc[i]['deg'])+ len(df1.iloc[i]['deg'])), "Degree")
    tup_skill = (df1.iloc[i]['jobpost'].find(df1.iloc[i]['RequiredQual']), int(df1.iloc[i]['jobpost'].find(df1.iloc[i]['RequiredQual'])+ len(df1.iloc[i]['RequiredQual'])), "Skill")
    tup_title = (df1.iloc[i]['jobpost'].find(df1.iloc[i]['Title']), int(df1.iloc[i]['jobpost'].find(df1.iloc[i]['Title'])+ len(df1.iloc[i]['Title'])), "Title")
    tup = []
    if df1.iloc[i]['jobpost'].find(df1.iloc[i]['deg'])!=-1:
        l1,l2,_ = tup_degree
        #print(df1.iloc[i]['jobpost'][l1:l2])
        if l1!=l2 and l1!=-1:
            tup.append(tup_degree)
    if df1.iloc[i]['jobpost'].find(df1.iloc[i]['RequiredQual'])!=-1:
        if len(df1.iloc[i]['RequiredQual'].split(df1.iloc[i]['deg']))>1:
            for j in df1.iloc[i]['RequiredQual'].split(df1.iloc[i]['deg']):
                    l1,l2 = df1.iloc[i]['jobpost'].find(j),df1.iloc[i]['jobpost'].find(j)+len(j)
                    if l1!=l2 and l1!=-1 and l2-l1 >2:
                        #print(df1.iloc[i]['jobpost'][l1:l2])
                        tup.append((l1,l2,'Skill'))
        else:
            l1,l2,_ = tup_skill
            if l1!=l2 and l1!=-1:
                tup.append(tup_skill)
    if df1.iloc[i]['jobpost'].find(df1.iloc[i]['Title'])!=-1:
        l1,l2,_ = tup_title
        if l1!=l2 and l1!=-1:
            #print(df1.iloc[i]['jobpost'][l1:l2])
            tup.append(tup_title)
    
    if len(tup)>=3: 
        list_ner.append((df1.iloc[i]['jobpost'], { 'entities': tup}))


with open('ner_train_data.pkl', 'wb') as fp:
    pickle.dump(list_ner, fp)
    
    

train_data = pickle.load(open('ner_train_data.pkl','rb'))

nlp_train_model = spacy.blank('en')

n_iter =10

def train_model(train_data):
    if 'ner' not in nlp_train_model.pipe_names:
        ner= nlp_train_model.create_pipe('ner')
        nlp_train_model.add_pipe(ner, last=True)
        
    for _, ann in train_data:
        for ent in ann['entities']:
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp_train_model.pipe_names if pipe != 'ner']
    with nlp_train_model.disable_pipes(*other_pipes):
        optimizer = nlp_train_model.begin_training()
        for itn in range(n_iter):
            print('Starting iter : '+str(itn))
            random.shuffle(train_data)
            losses = {}
            
            for text, an in train_data:
                try:
                    if len(text)>0:
                        nlp_train_model.update([text], [an], drop=0.2, sgd=optimizer, losses=losses)
                except Exception as e:
                    print(e)
                    pass
            print("Losses", losses)
            
train_model(train_data[:500])



nlp_train_model.to_disk('nlp_model')

nlp_ner_model = spacy.load('nlp_model')



df1 = df.copy()
df1 = df1[df1['exp'].notnull()].copy()


list_ner = []
for i in range(len(df1['jobpost'])):
    for j in df1.iloc[i]['exp']:
        tup_exp = (df1.iloc[i]['jobpost'].find(j), int(df1.iloc[i]['jobpost'].find(j)+ len(j)), "Experience")
        l1,l2,_ = tup_exp
        if l1!=l2 and l1!=-1:
            list_ner.append((df1.iloc[i]['jobpost'], { 'entities': [tup_exp]}))
        
    

with open('ner_experience_data.pkl', 'wb') as fp:
    pickle.dump(list_ner, fp)
    
    

train_data_exp = pickle.load(open('ner_experience_data.pkl','rb'))

nlp_exp_train_model = spacy.blank('en')

n_iter =10

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
            
train_model_exp(train_data_exp[:500])



nlp_exp_train_model.to_disk('nlp_exp_model')

nlp_exp_model = spacy.load('nlp_exp_model')


skill_train = pd.read_csv(r'D:\Intelligent Systems\Practical Language Processing\JD-Parser\skill_train.csv')


X = skill_train.phrase
y = skill_train.skill

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = .2,
                                                    random_state = 42,
                                                    shuffle = True
                                                    )



############# word embedding

# define documents
vocab_size = len(skill_train.phrase)
encoded_docs = [one_hot(d, vocab_size) for d in (X_train)]
max_length = max([len(s.split()) for s in (skill_train.phrase)])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

encoded_docs_test = [one_hot(d, vocab_size) for d in (X_test)]
max_length = max([len(s.split()) for s in (skill_train.phrase)])
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')

# define the model_emb
model_emb = Sequential()
model_emb.add(Embedding(10000, 8, input_length=max_length))
model_emb.add(Flatten())
model_emb.add(Dense(1, activation='sigmoid'))
# compile the model_emb
model_emb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model_emb
print(model_emb.summary())
# fit the model_emb
model_emb.fit(padded_docs, y_train, epochs=100, verbose=0)

loss, accuracy = model_emb.evaluate(padded_docs_test,y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print(classification_report(y_test, model_emb.predict_classes(padded_docs_test)))

model_emb.save("model_emb.h5")


######################## embedding + conv


from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

# define model_conv
model_conv = Sequential()
model_conv.add(Embedding(vocab_size, 100, input_length=max_length))
model_conv.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
model_conv.add(MaxPooling1D(pool_size=1))
model_conv.add(Flatten())
model_conv.add(Dense(10, activation='relu'))
model_conv.add(Dense(1, activation='sigmoid'))
print(model_conv.summary())


# compile network
model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model_conv.fit(padded_docs, y_train, epochs=75, verbose=2)

# evaluate
loss, acc = model_conv.evaluate(padded_docs_test,y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))
print(classification_report(y_test, model_conv.predict_classes(padded_docs_test)))

model_conv.save("model_conv.h5")



################################ glove 

from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
t = Tokenizer(oov_token = True)
t.fit_on_texts((X_train))
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences((X_train))
# pad documents to a max length 
max_length = max([len(s.split()) for s in (skill_train.phrase)])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open(r'glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))



# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# define model_glove
model_glove = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model_glove.add(e)
model_glove.add(Flatten())
model_glove.add(Dense(1, activation='sigmoid'))
# compile the model_glove
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model_glove
print(model_glove.summary())
# fit the model_glove
model_glove.fit(padded_docs, y_train, epochs=50, verbose=1)



# integer encode the documents
encoded_docs_test = t.texts_to_sequences((X_test))
# print(encoded_docs_test)
# pad documents 
#max_length = 9
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')



# evaluate the model_glove
loss, acc = model_glove.evaluate(padded_docs_test, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))
print(classification_report(y_test, model_glove.predict_classes(padded_docs_test)))

model_glove.save("model_glove.h5")

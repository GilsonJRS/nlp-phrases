import dill
import spacy
import string
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing
from spacy.lang.pt import Portuguese
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#data
data = pd.read_csv("./data/dataset.csv")
classes = ['finanças', 'educação', 'indústrias', 'varejo', 'orgão público']

#the lg core is more complex and precise, but bigger
nlp = spacy.load('pt_core_news_sm')
#nlp = spacy.load('pt_core_news_lg')
ponctuations = string.punctuation

#function to tokenize the sentences
#this function remove the stop words
def tokenizer(text):
  text = text.lower()
  tokens = nlp(text)
 
  tokens = [word.lemma_.strip() if word.lemma_ != "-PRON-" else word for word in tokens]
  tokens = [word for word in tokens if word not in STOP_WORDS and str(word) not in ponctuations]
  return tokens
  
#processing data to fit the model
mlb = preprocessing.MultiLabelBinarizer()

count_word_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer)

X = data.sentence
y = data.category.str.split(',')
mlb.fit([classes])
y = mlb.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

#Using a Logistic Regression model, with sklearn wraper to multilabel
model = MultiOutputClassifier(LogisticRegression(class_weight='balanced'), n_jobs = 4)
pipe_logistic = Pipeline([("vectorizer", count_word_vectorizer),
                 ("classifier", model)])
pipe_logistic.fit(X_train, y_train)

#print some statistics to see model performance
predicted = pipe_logistic.predict(X_test)
print('Precision:',metrics.precision_score(y_test, predicted, average='weighted', zero_division=0))
print('Recall:',metrics.recall_score(y_test, predicted, average='weighted',zero_division=0))
print('F1:',metrics.f1_score(y_test, predicted, average='weighted'))

#save the model
dump(pipe_logistic['classifier'], 'model.pkl')
dill.dump(pipe_logistic['vectorizer'], open("vectorizer.pkl", 'wb'))


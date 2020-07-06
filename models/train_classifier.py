# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    return X,Y,category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




def build_model():
    pipeline  = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    '''parameters = {
      'vect__max_df': (0.5, 0.75),
    # 'vect__max_features': (None, 5000, 10000, 50000),
     # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
       # 'clf__estimator__n_jobs': (None,1),
      # 'clf__estimator__n_estimators': (100,200,500)
        
    }
    cv = GridSearchCV(pipeline, parameters, verbose=1)'''
    return pipeline  #cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred= model.predict(X_test)
    
    for i in range(36):
      print('Category name:'+category_names[i])
      print(classification_report(Y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
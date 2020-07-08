import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import send_file
from flask import Flask, url_for
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
import re
from collections import Counter
from itertools import islice

app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    stopwords_ = stopwords.words("english")
    clean_tokens = [word for word in clean_tokens if word not in stopwords_]

    return clean_tokens



    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Extract categories informations from dataframe
    cat_counts = df.iloc[:,4:].sum()
    cat_names = list(df.columns[4:])

    words=[]
    for word in df.message[:300].values:
        words.extend(tokenize(word))

    word_count_dict = Counter(words)
    dict= {k: v for k, v in sorted(word_count_dict.items(), key=lambda item: item[1],reverse=True)}
    def get_list(dict):
        return dict.keys()
    def get_values(dict):
        return dict.values()

    words = list(get_list(dict))[:15]
    count = list(get_values(dict))[:15]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count
                )
            ],

            'layout': {
                'title': 'Most repeated words out of 300 messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results'''
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )




def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import praw
import json
import pickle
import numpy as np
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = pickle.load(open('model.pkl','rb'))

def preprocessing(ent):

    ent = str(ent)

    ent = ent.lower()
    ent = ent.replace("\n", "")
    ent = re.sub(r'^0-9a-z #+_', '', ent) #Removing digits
    ent = re.sub(r'^https?:\/\/.*[\r\n]*', '', ent) #Removing URLs

    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

    for char in ent:
        if char in punctuations:
            ent = ent.replace(char, " ")

    ent = ent.strip()

    return ent

##

def remstopwords(string):

    string = str(string)

    wordsCorpus = set(nltk.corpus.words.words())
    wordsCorpus.add('.')

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(string)

    words = [word for word in tokens if word not in stop_words]
    words = [word for word in words if word in wordsCorpus or not word.isalpha]

    string = ' '.join(word for word in words)
    string = string.strip()

    return string

##

def lemmatize(string):

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(string)
    for ind in range(len(tokens)):

        tokens[ind] = lemmatizer.lemmatize(tokens[ind])

    string = ' '.join(word for word in tokens)

    return string

##

def data(url):
    reddit = praw.Reddit(client_id = 'gb2ZbwyyADJBbw', client_secret = 'nGdE_EtfK8pM20dGy369ZmOJxOg', user_agent = 'SubScraper', username = 'KaranjotSinghV', password = 'jogindersinghvilkhu')
    post = reddit.submission(url = url)

    data = {}
    data["title"] = str(post.title)
    data["url"] = str(post.url)
    data["body"] = str(post.selftext)

    post.comments.replace_more(limit=None)
    comment = ''
    count = 0
    for top_level_comment in post.comments:
        comment = comment + ' ' + top_level_comment.body
        count+=1
        if(count > 10):
            break
    data["comment"] = str(comment)
    flairRetrieved = post.link_flair_text

    data['title'] = preprocessing(str(data['title']))
    data['title'] = remstopwords(str(data['title']))
    data['title'] = lemmatize(str(data['title']))

    data['body'] = preprocessing(str(data['body']))
    data['body'] = remstopwords(str(data['body']))
    data['body'] = lemmatize(str(data['body']))

    data['comment'] = preprocessing(str(data['comment']))
    data['comment'] = remstopwords(str(data['comment']))
    data['comment'] = lemmatize(str(data['comment']))

    combined_features = [data["title"] + data["comment"] + data["body"] + data["url"]]
    return combined_features, flairRetrieved

##

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predictor():

    if request.method == 'POST':

        url = request.form['url']
        combined_features, flairRetrieved = data(url)
        flair = model.predict(combined_features)
        return render_template('index.html', predict = flair[0], flairRetrieved = flairRetrieved)
    return render_template('form.html')

##

@app.route("/automated_testing",methods=['GET', 'POST'])
def test():

    if request.method == 'POST':

        file = request.files["upload_file"]
        texts = file.read()
        texts = str(texts.decode('utf-8'))
        links = texts.split('\n')
        pred = {}
        for ind in range(len(links)-1):
            combined_features, flairRetrieved = data(links[ind])
            pred[links[ind]] = str(model.predict(combined_features))
        return jsonify(pred)
    else:
        return("Send a .txt file containing valid Reddit URLs via a POST request to this end-point for automated testing!")

##

if __name__ == '__main__':
	app.run(port = 5500)

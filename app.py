import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import praw
import json
from helper import *
import pickle

model = pickle.load(open('LR-PUSHSHIFT.pkl','rb'))

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
        files = request.files["upload_file"]
        files.save(os.path.join("sample", "submission.txt"))
        with open("sample/submission.txt") as f:
            texts = f.read()

        links = texts.split('\n')
        pred = {}
        for ind in range(len(links) - 1):
            combined_features, flairRetrieved = data(links[ind])
            pred[links[ind]] = model.predict(combined_features)[0]
        with open("sample/predictions.json", 'w') as fp:
            json.dump(pred, fp)
        return redirect(url_for("return_file"))

    return render_template('upload.html')

##
@app.route('/downloadfile')
def return_file():
    try:
        return render_template('download.html')
    except Exception as e:
        return str(e)

##

@app.route('/downloadjson')
def return_json():
    try:
        return send_file('sample/predictions.json', as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
	app.run(port = 5500)

from flask import render_template
from flask import jsonify
from flask import abort
from flask import make_response
from flask import request
from flask import url_for
from app import app
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import load

import spacy
import subprocess

import json
import os
import random

import GPUtil
import tensorflow as tf
from bert_serving.client import ConcurrentBertClient
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
import en_core_web_sm

texts = ('aaa', 'bbb')
text = []
tf.logging.set_verbosity(tf.logging.INFO)

train_fp = ['xxx.txt']
eval_fp = ['zzz.txt']
batch_size = 32
#num_parallel_calls = 1

num_parallel_calls = 4
num_concurrent_clients = num_parallel_calls * 2

#bc = BertClient()
bc = ConcurrentBertClient()

def get_encodes(x):
    print(x)
    print(text)
    features = bc.encode(text)
    # randomly choose a label
    labels = [['0']]
    return features, labels


def sent_vectorizer(sent, model):
    sent_vec =[]
    print(sent)
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    return np.asarray(sent_vec) / numw

def predict(txt):
#    print('!')
    model = load('/var/www/TestApp/ft.joblib')
    classifier2 = load('/var/www/TestApp/model.joblib')
    #print('model loaded')
    sentenceX = txt.lower().split(' ')

    VV=[]
    VV.append(sent_vectorizer(sentenceX, model))
    #return txt
    res= classifier2.predict(VV)
    print(res[0])
    return jsonify(res[0])

def predict_bert(txt, return_probs, tx, tx_prob):
    text.append(txt)
    config = tf.ConfigProto()
    run_config = RunConfig(model_dir='/home/ubuntu/dnn_models/', session_config=config, save_checkpoints_steps=1000)

    estimator = DNNClassifier(
        hidden_units=[512],
        feature_columns=[tf.feature_column.numeric_column('feature', shape=(768,))],
        n_classes=2,
        config=run_config,
        label_vocabulary=['0','1'],
        dropout=0.1)

    input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                           .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'), num_parallel_calls=num_parallel_calls)
                           .map(lambda x, y: ({'feature': x}, y)))

    print('!!!')
    print(estimator.latest_checkpoint())

    predictions = estimator.predict(input_fn=lambda: input_fn(train_fp), predict_keys=['probabilities','classes'])
    #res = predictions['classes']
    res = []
    res_p = []
    for p in predictions:
      #  print(res)
     #   print(p['classes'])
        res.append(p['classes'])
        res_p.append(p['probabilities'])
#    print(res)
    print(res[0])
    print(res_p[0])
    text.clear()
    labl = int(res[0][0].decode('utf-8'))
    prob = res_p[0][0]
    print(prob)
    if(return_probs):
        return jsonify(str(prob))
    else:
        if(tx):
            if(tx_prob):
                return labl, prob
            else:
                return labl
        else:
            return jsonify(labl)



def retrain_model():
#    subprocess.run("/home/ubuntu/bert/dnn_train.py", shell=True)
    #subprocess.call(["sudo","python3","/home/ubuntu/bert/dnn_train.py"], cwd="/home/ubuntu/bert")
    subprocess.Popen(["python3", "/home/ubuntu/bert/dnn_train.py"], cwd="/home/ubuntu/bert")
    return 'r'

@app.route('/')
def home():
    return "Demo Webapp for the ReqExp project"

@app.route('/show', methods=['GET'])
def show():
    return texts

@app.route('/prob', methods=['POST'])
def classify_text():
    label =  predict_bert(request.json['text'], True, False, False)
    return label


@app.route('/classify', methods=['POST'])
def classify_bert():
    label =  predict_bert(request.json['text'], False, False, False)
    return label

@app.route('/addtrain', methods=['POST'])
def add_text():
    train_arr = request.json['trainset']
    i = 1
    f= open("/home/ubuntu/bert/zzz_total.txt","w+")
    for r in train_arr:
        f.write(str(i)+'\t'+ r[1]+ '\ta\t'+ r[0]+'\n')
        i = i + 1
    f.close()
    return 'added'

@app.route('/retrain', methods=['POST'])
def retrain():
    print('retrain_started')
    retrain_model()
    return 'done'

@app.route('/status', methods=['GET'])
def get_retrain_status():
    f = open("/home/ubuntu/bert/status.txt","r")
    res_st = f.readline()
    f.close()
    return res_st

@app.route('/texts', methods=['POST'])
def classify_sentences():
    #nlp = spacy.load('en')
    #nlp = spacy.load('en_core_web_sm')
    print('in')
    arr = []
    data_text = request.json['text']
    doc = nlp(data_text)
    for sent in doc.sents:
        s = ''.join(token.string for token in sent)
        print(s)
        label = predict_bert(s.lower(), False, True, False)
        arr.append([s, str(label)])
    return jsonify(arr)

@app.route('/textsprob', methods=['POST'])
def classify_sentences_prob():
    #nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
    arr = []
    data_text = request.json['text']
    doc = nlp(data_text)
    for sent in doc.sents:
        s = ''.join(token.string for token in sent)
        print('prob', s)
        label, prob = predict_bert(s.lower(), False, True, True)
        arr.append([s, str(label), str(prob)])
    return jsonify(arr)


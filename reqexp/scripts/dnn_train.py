#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# solving chinese law-article classification problem: https://github.com/thunlp/CAIL/blob/master/README_en.md

import json
import os
import random

import GPUtil
import tensorflow as tf
from bert_serving.client import ConcurrentBertClient
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

#os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
#tf.logging.set_verbosity(tf.logging.INFO)

#train_fp = ['xxx.txt']

status_file = open("/home/ubuntu/bert/status.txt","w+")
status_file.write("training\n")
status_file.close()

train_fp = ['zzz_total.txt']
eval_fp = ['zzz_total.txt']
batch_size = 64
num_parallel_calls = 2
num_concurrent_clients = num_parallel_calls * 2  # should be at least greater than `num_parallel_calls`

bc = ConcurrentBertClient()

def get_encodes(x):
    # x is `batch_size` of lines, each of which is a json object
#    print(x)
    l = []
    txt = []
    for s in x:
        strx = s.decode('utf-8')
        vals = strx.split('\t')
        l.append([vals[1]])
        txt.append(vals[3].lower())
#    print('!', txt)
#    print('#', l)
    features = bc.encode(txt)
    # randomly choose a label
    labels = l
    #print(features, labels)
    return features, labels


config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
run_config = RunConfig(model_dir='/home/ubuntu/dnn_models/',
                       session_config=config,
                       save_checkpoints_steps=10)

estimator = DNNClassifier(
    hidden_units=[512],
    feature_columns=[tf.feature_column.numeric_column('feature', shape=(768,))],
    n_classes=2,
    config=run_config,
    label_vocabulary=['0','1'],
    dropout=0.1)

input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                       .shuffle(20000)
                       .repeat()
                       .batch(batch_size)
                       .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'), num_parallel_calls=num_parallel_calls)
                       .map(lambda x, y: ({'feature': x}, y))
                       .prefetch(20))

estimator.train(input_fn=lambda: input_fn(train_fp), steps=10)

status_file = open("/home/ubuntu/bert/status.txt","w+")
status_file.write("ready\n")
status_file.close()

#train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp))
#eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp), throttle_secs=0)
#train_and_evaluate(estimator, train_spec, eval_spec)

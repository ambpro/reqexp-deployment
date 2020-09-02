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
from bert_serving.client import BertClient
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

#os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
tf.logging.set_verbosity(tf.logging.INFO)

train_fp = ['zzz.txt']
eval_fp = ['zzz.txt']

batch_size = 128
num_parallel_calls = 1
#num_concurrent_clients = num_parallel_calls * 2  # should be at least greater than `num_parallel_calls`

bc = BertClient()

def get_encodes(x):
    # x is `batch_size` of lines, each of which is a json object
    #text = ['system should be able to work','can perform the same inventory management functions as a receiving associate plus change item properties','mhi images would include cctv and still images']
    
    features = bc.encode(text)
    # randomly choose a label
    labels = [['0']]
    return features, labels


config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
run_config = RunConfig(model_dir='/tmp/',
                       session_config=config,
                       save_checkpoints_steps=1000)

estimator = DNNClassifier(
    hidden_units=[512],
    feature_columns=[tf.feature_column.numeric_column('feature', shape=(768,))],
    n_classes=2,
    config=run_config,
    label_vocabulary=['0','1'],
    dropout=0.1)

input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                       # .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                       # .batch(batch_size)
                       .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'), num_parallel_calls=num_parallel_calls)
                       .map(lambda x, y: ({'feature': x}, y)))
                       # .prefetch(20))

input_fn2 = lambda fp: (tf.data.TextLineDataset(fp)
                       .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                       .batch(batch_size)
                       .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'),
                            num_parallel_calls=num_parallel_calls)
                       .map(lambda x, y: ({'feature': x}, y))
                       .prefetch(20))

#train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp))
#eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp))
#train_and_evaluate(estimator, train_spec, eval_spec)
print('!!!')
print(estimator.latest_checkpoint())

#print(estimator)

#estimator.train(input_fn=lambda: input_fn(train_fp))
predictions = estimator.predict(input_fn=lambda: input_fn(train_fp), predict_keys=['probabilities','classes'])
for p in predictions:
    print(p)

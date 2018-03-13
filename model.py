from loader import load_data, seq_classes, seq_tokens
import json,random,os
import loader

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

from data_helpers import num_tokens, num_classes, pad_token



def last_relevant(output, length, time_major=False):
  batch_size = tf.shape(output)[(1 if time_major else 0)]
  max_length = tf.shape(output)[(0 if time_major else 1)]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

def lrelu(x, alpha=0.1):
  return tf.maximum(x, alpha * x)

# model params
num_epochs=12
learning_rate=0.0001
batch_size=16
fully_connected_hidden_units=64
num_units = 64
cell_type = 'GRU'
dropout_rate=0.2

to_restore=True
do_training = False

def build_graph(embedding_size=None, num_conv_filters=None, rnn_depth=2):

    # create variables
    W_h1 = tf.get_variable('W_h1', [num_units*2,fully_connected_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b_h1 = tf.get_variable('b_h1', [fully_connected_hidden_units], dtype=tf.float32)
    W_out = tf.get_variable('W_out', [fully_connected_hidden_units,num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.get_variable('b_out', [num_classes], dtype=tf.float32)

    # placeholders
    x = tf.placeholder(tf.int64, [None, None])
    x_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_token) ,tf.int32),axis=1)
    y = tf.placeholder(tf.int64, [None])
    dropout_active = tf.placeholder_with_default(False,(), "dropout_active")



    if embedding_size is not None:
        embedding_encoder = tf.get_variable("encoder_embed", [num_tokens-1, embedding_size], initializer=tf.orthogonal_initializer())
        embedding_encoder = embedding_encoder/tf.norm(embedding_encoder, axis=1, keep_dims=True)
        embedding_encoder = tf.concat([tf.zeros([1,embedding_size]), embedding_encoder],0)

        x_embedded = tf.nn.embedding_lookup(
            embedding_encoder, x)
    else:
        embedding_encoder = tf.get_variable("encoder_embed", [num_tokens-1, 1], initializer=tf.orthogonal_initializer())
        x_embedded = tf.one_hot(x, num_tokens)

    if num_conv_filters is not None:
        x_conv = tf.layers.conv2d(
            tf.expand_dims(x_embedded,-1),
            num_conv_filters,
            (3,3),
            padding='same')
        x_conv2 = tf.layers.conv2d(
            x_conv,
            1,
            (3,3),
             padding='same')
        x_out = tf.squeeze(x_conv2, -1) + x_embedded
    else:
        x_out = x_embedded


    # define the RNN cell
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units) if cell_type=='GRU' else tf.contrib.rnn.BasicLSTMCell(num_units) for c in range(rnn_depth)])
    cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units) if cell_type=='GRU' else tf.contrib.rnn.BasicLSTMCell(num_units) for c in range(rnn_depth)])

    # unroll the RNN
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
      cell, cell_bw, x_out, x_len, dtype=tf.float32)

    # get last output, pass through relu
    last_output = lrelu(tf.layers.dropout(last_relevant(tf.concat(outputs, 2), length=x_len), rate=dropout_rate, training=dropout_active))

    # pass through relu and FC layers
    h1 = lrelu(tf.layers.dropout(tf.matmul(last_output, W_h1) + b_h1, rate=dropout_rate, training=dropout_active))
    logits = tf.matmul(h1, W_out) + b_out

    y_hat = tf.nn.softmax(logits)

    # cross entropy loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=logits))

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, 5)

    # Optimization
    optimizer = tf.train.AdamOptimizer(0.001)
    opt = optimizer.apply_gradients(
        zip(clipped_gradients, params))

    y_hat_oh = tf.argmax(y_hat,axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_hat_oh,y), tf.float32))

    return x,y,opt, accuracy, y_hat, loss, embedding_encoder,dropout_active

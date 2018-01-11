# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# train_model.py

# This script trains the model

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import tensorflow as tf # change this to gpu
import matplotlib.pyplot as plt
import numpy as np

import load_data
import model

# training parameters
learning_rate = 0.001
epochs = 100
batch_size = 100
display_step = 5

# net parameters
input_dim = 20
# hidden_dim = [256, 64, 16]
hidden_dim = [14, 8]
classes = 2
dropout = 0.25 # probability to drop a unit

# load data
x_train, x_test, y_train, y_test = load_data.load()
print(y_train.shape)
# inputs
x = tf.placeholder("float", [None, input_dim])
y = tf.placeholder("float", [None, classes])

# model
output = model.kd_model(x, hidden_dim, classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()


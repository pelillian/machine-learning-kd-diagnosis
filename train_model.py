# -------------------------------------------------------------------------------------------------
# MACHINE LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# train_model.py

# Functions for loading data

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import tensorflow as tf # change this to gpu
import matplotlib.pyplot as plt
import numpy as np

import load_data
import model

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
num_input = 14
num_hidden_1 = 256
num_hidden_2 = 32
num_classes = 2
dropout = 0.25 # probability to drop a unit

x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, num_classes])


# -------------------------------------------------------------------------------------------------
# MACHINE LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# model.py

# Function for defining the model

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import tensorflow as tf # change this to gpu
import numpy as np

# input
def model(x, num_hidden, num_classes):
	fc1 = tf.contrib.layers.fully_connected(x, num_hidden[0])
	fc2 = tf.contrib.layers.fully_connected(fc1, num_hidden[1])
	fc3 = tf.contrib.layers.fully_connected(fc2, num_hidden[2])
	fc4 = tf.contrib.layers.fully_connected(fc3, num_classes, activation_fn=None)
	return fc4
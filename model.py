# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# model.py

# Function for defining the model

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import tensorflow as tf # change this to gpu
import numpy as np

# input
def kd_model(x, hidden_dim, classes):
	fc = x
	for layer_size in hidden_dim:
		fc = tf.contrib.layers.fully_connected(fc, layer_size)

	fc = tf.contrib.layers.fully_connected(fc, classes, activation_fn=None)
	
	return fc
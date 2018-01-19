# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# train_model.py

# This script sets up and trains the model

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import tensorflow as tf # change this to gpu
import matplotlib.pyplot as plt
import numpy as np

import load_data
import model

# training parameters
learning_rate = 0.001
epochs = 200
batch_size = 100
display_step = 5

# net parameters
input_dim = 20
hidden_dim = [40, 30, 20, 14, 9, 5]
classes = 2
dropout = 0.95 # probability to keep a unit

# load data
x_train, x_test, y_train, y_test = load_data.load()

# inputs
x = tf.placeholder("float", [None, input_dim])
y = tf.placeholder("float", [None, classes])
keep_prob = tf.placeholder(tf.float32)

# model
output = model.kd_model(x, hidden_dim, classes, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# run session: train
with tf.Session() as sess:
	# initialize vars
	sess.run(init)

	# start training
	for epoch in range(epochs):
		avg_cost = 0.
		num_batches = len(x_train)//batch_size
		batches_array_x = np.array_split(x_train, num_batches)
		batches_array_y = np.array_split(y_train, num_batches)

		for i in range(num_batches):
			x_batch, y_batch = batches_array_x[i], batches_array_y[i]

			batch_cost = sess.run([opt, cost], feed_dict={x: x_batch, y: y_batch, keep_prob: dropout})[1]

			avg_cost += batch_cost

		avg_cost /= num_batches

		if epoch % display_step == 0:
			print('Epoch', epoch + 1, ' cost', avg_cost)

	# Run quick test
	end_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(end_pred, "float"))

	test_accuracy = accuracy.eval({x: x_test, y: y_test, keep_prob: 1})

	print("Accuracy", test_accuracy)

# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# model.py

# This script defines and trains the deep model

# Peter Lillian
# -------------------------------------------------------------------------------------------------

# hide tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf # change this to gpu
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

class DeepKDModel:
	def __init__(self, learning_rate=0.001, epochs=200, batch_size=100, display_step=10,
			classes=2, dropout=0.75, verbose=False):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.display_step = display_step
		self.classes = classes
		self.dropout = dropout # probability to keep a unit
		self.verbose = verbose

	def kd_model(self, x, hidden_dim, classes, keep_prob):
		fc = x
		for layer_size in hidden_dim:
			fc = tf.contrib.layers.fully_connected(fc, layer_size)
			fc = tf.nn.dropout(fc, keep_prob)

		fc = tf.contrib.layers.fully_connected(fc, classes, activation_fn=None)
		
		return fc

	def train_test(self, x_train, x_test, y_train, y_test):
		self.train(x_train, y_train)
		return self.test(x_test, y_test)

	def train(self, x_train, y_train):
		# net parameters
		input_dim = len(x_train[1, :])
		hidden_dim = [input_dim * 2, int(input_dim * 1.5), input_dim, int(input_dim * 0.7),
			input_dim // 2, input_dim // 4]

		# preprocessing
		self.scaler = preprocessing.StandardScaler().fit(x_train)
		x_train = self.scaler.transform(x_train)

		# inputs
		self.x = tf.placeholder("float", [None, input_dim])
		self.y = tf.placeholder("float", [None, self.classes])
		self.keep_prob = tf.placeholder(tf.float32)

		# model
		self.model = self.kd_model(self.x, hidden_dim, self.classes, self.keep_prob)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.model, labels=self.y
		))
		opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

		init = tf.global_variables_initializer()

		# run session: train
		with tf.Session() as sess:
			# initialize vars
			sess.run(init)

			saver = tf.train.Saver()

			global_step = 0

			# start training
			for epoch in range(self.epochs):
				avg_cost = 0.
				num_batches = len(x_train)//self.batch_size
				batches_array_x = np.array_split(x_train, num_batches)
				batches_array_y = np.array_split(y_train, num_batches)

				for i in range(num_batches):
					x_batch, y_batch = batches_array_x[i], batches_array_y[i]

					batch_cost = sess.run([opt, cost], feed_dict={
						self.x: x_batch, self.y: y_batch, self.keep_prob: self.dropout
					})[1]

					avg_cost += batch_cost
					global_step += 1

				avg_cost /= num_batches

				if self.verbose and epoch % self.display_step == 0:
					print('Epoch', epoch + 1, ' cost', avg_cost)

			saver.save(sess, './deepnet/deep_kd_model')

	def test(self, x_test, y_test):

		# preprocessing
		x_test = self.scaler.transform(x_test)

		# set up tf
		y_pred = tf.argmax(self.model, 1)
		# end_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
		# accuracy = tf.reduce_mean(tf.cast(end_pred, "float"))

		with tf.Session() as sess:
			saver = tf.train.import_meta_graph('./deepnet/deep_kd_model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('./deepnet/'))
			test_results = y_pred.eval({self.x: x_test, self.y: y_test, self.keep_prob: 1})

			# if self.verbose:
			# 	print("Accuracy", test_accuracy)

			return test_results

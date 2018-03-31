# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# model.py

# This script defines and trains the deep model

# Peter Lillian, Lucas Hu
# -------------------------------------------------------------------------------------------------

# hide tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf # change this to gpu
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import fbeta_score
from skopt import gp_minimize

class DeepKDModel:
	def __init__(self, 
			num_hidden_layers=3, num_nodes_initial=50, num_nodes_scaling_factor=0.5,
			learning_rate=0.001, epochs=400, batch_size=100, display_step=10,
			classes=2, dropout=0.75, verbose=False):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.display_step = display_step
		self.classes = classes
		self.dropout = dropout # probability to keep a unit
		self.verbose = verbose

		# Architecture hyperparameters
			# Number of layers
			# Num. nodes in 1st hidden layer
			# Scaling factor (downscale number of nodes in each hidden layer)
		self.num_hidden_layers = num_hidden_layers
		self.num_nodes_initial = num_nodes_initial
		self.num_nodes_scaling_factor = num_nodes_scaling_factor


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
		tf.reset_default_graph()

		# onehot
		y_train = y_train.astype(int)
		y_train = np.eye(np.max(y_train) + 1)[y_train]

		# net parameters
		input_dim = len(x_train[1, :])

		# List of values representing the num. nodes in each hidden layer
			# Num. nodes decreases by num_nodes_scaling_factor in each successive layer
		hidden_dim = [int(self.num_nodes_initial * (self.num_nodes_scaling_factor**i)) \
			for i in range(self.num_hidden_layers)]

		if self.verbose:
			print('Initialized MLP with hidden layer sizes: ', hidden_dim)

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

	# Returns binary vector of predictions from test set (1 = KD, 0 = FC)
	def test(self, x_test, y_test):
		# preprocessing
		x_test = self.scaler.transform(x_test)
		# onehot
		y_test = y_test.astype(int)
		y_test = np.eye(np.max(y_test) + 1)[y_test]

		# set up tf
		y_pred = tf.argmax(self.model, axis=1)
		# end_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
		# accuracy = tf.reduce_mean(tf.cast(end_pred, "float"))

		with tf.Session() as sess:
			saver = tf.train.import_meta_graph('./deepnet/deep_kd_model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('./deepnet/'))
			test_results = y_pred.eval({self.x: x_test, self.y: y_test, self.keep_prob: 1})

			# if self.verbose:
			# 	print("Accuracy", test_accuracy)

			return test_results # binary vector of test results


	# Weighted score of precision/recall
		# See: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
	def fbeta(self, x_test, y_test, beta=1):
		y_test_binary = np.argmax(y_test, axis=1) # un-one hot y_test

		# preprocessing
		x_test = self.scaler.transform(x_test)

		# set up tf
		y_pred = tf.argmax(self.model, axis=1)
		# end_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
		# accuracy = tf.reduce_mean(tf.cast(end_pred, "float"))

		with tf.Session() as sess:
			saver = tf.train.import_meta_graph('./deepnet/deep_kd_model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('./deepnet/'))
			y_pred_binary = y_pred.eval({self.x: x_test, self.y: y_test, self.keep_prob: 1})

			# if self.verbose:
			# 	print("Accuracy", test_accuracy)

			# y_pred_binary = np.where(test_results >= 0.5, 1, 0) # threshold predictions at 0.5
			return fbeta_score(y_test_binary, y_pred_binary, beta)


	# Optimize model hyperparameters based on fbeta_score, save optimal parameters in member vars
	def optimize_hyperparameters(self, x_train, x_test, y_train, y_test, beta=1):
		# Optimization objective for skopt: returns negated fbeta score
		# Input: tuple of hyperparameters
		def skopt_objective(params):
			# Unpack parameters
			num_hidden_layers, num_nodes_initial, num_nodes_scaling_factor, \
				epochs, learning_rate, batch_size, dropout = params

			# Set parameters
			self.num_hidden_layers = num_hidden_layers
			self.num_nodes_initial = num_nodes_initial
			self.num_nodes_scaling_factor = num_nodes_scaling_factor
			self.epochs = epochs
			self.learning_rate = learning_rate
			self.batch_size = batch_size
			self.dropout = dropout # probability to keep a unit

			# Train model using inputted hyperparameters
			self.train(x_train, y_train)

			# Return negated fbeta_score (minimize negative fbeta --> maximize fbeta)
			return -self.fbeta(x_test, y_test, beta)

		# Define hyperparameter space
		hyperparam_space = [
			(1, 5), # num_hidden_layers
			(50, 250), # num nodes initial
			(0.5, 1.0), # num nodes scaling factor
			(1, 500), # epochs
			(10**-6, 10**-1, 'log-uniform'), # learning_rate
			(16, 128), # batch size
			(0.5, 1.0) # dropout
			]

		# Call skopt to run smart "grid search"
		opt_results = gp_minimize(skopt_objective, hyperparam_space,
						n_calls=10,
						random_state=0,
						verbose=self.verbose)
		# Unpack results
		optimal_hyperparams = opt_results.x
		optimal_score = opt_results.fun

		opt_num_hidden_layers = optimal_hyperparams[0]
		opt_num_nodes_initial = optimal_hyperparams[1]
		opt_num_nodes_scaling_factor = optimal_hyperparams[2]
		opt_epochs = optimal_hyperparams[3]
		opt_learning_rate = optimal_hyperparams[4]
		opt_batch_size = optimal_hyperparams[5]
		opt_dropout = optimal_hyperparams[6]

		# Print hyperparameter optimization results
		if self.verbose:
			print()
			print('----- HYPERPARAMETER OPTIMIZATION RESULTS -----')
			print('Optimal fbeta score: ', optimal_score)
			print('Optimal num_hidden_layers: ', opt_num_hidden_layers)
			print('Optimal num_nodes_initial: ', opt_num_nodes_initial)
			print('Optimal num_nodes_scaling_factor: ', opt_num_nodes_scaling_factor)
			print('Optimal epochs: ', opt_epochs)
			print('Optimal learning_rate: ', opt_learning_rate)
			print('Optimal batch_size: ', opt_batch_size)
			print('Optimal dropout (keep_prob): ', opt_dropout)

		# Update model hyperparameter member vars
		self.num_hidden_layers = opt_num_hidden_layers
		self.num_nodes_initial = opt_num_nodes_initial
		self.num_nodes_scaling_factor = opt_num_nodes_scaling_factor
		self.epochs = opt_epochs
		self.learning_rate = opt_learning_rate
		self.batch_size = opt_batch_size
		self.dropout = opt_dropout

		# Train 1 last time using optimal hyperparams
		print()
		print('Re-training with optimal hyperparameters...')
		self.train(x_train, y_train)




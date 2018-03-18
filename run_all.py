# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from deepnet.deep_model import DeepKDModel
from preprocess import load_data

def compute_stats(y_pred, y_test):
	y_results = np.column_stack((y_test[:, 1], y_pred))
	y_arr = np.dtype((np.void, y_results.dtype.itemsize * y_results.shape[1]))
	contigview = np.ascontiguousarray(y_results).view(y_arr)
	return np.unique(contigview, return_counts=True)[1]

def explain_stats(stats):
	print("FC Classified as FC: " + str(stats[0]))
	print("FC Classified as KD: " + str(stats[1]))
	print("KD Classified as FC: " + str(stats[2]))
	print("KD Classified as KD: " + str(stats[3]))


# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=True, fill_mode='mean')

# test deepmodel
deep = DeepKDModel()
deep.train(x_train, y_train)
deep_y_pred = deep.test(x_test, y_test)
deep_stats = compute_stats(deep_y_pred, y_test)

explain_stats(deep_stats)

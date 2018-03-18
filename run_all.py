# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from deepnet.deep_model import DeepKDModel
from preprocess import load_data

def compute_stats(y_test, y_pred):
	y_results = np.column_stack((y_test[:, 1], y_pred))

	dt = np.dtype((np.void, y_results.dtype.itemsize * y_results.shape[1]))
	b = np.ascontiguousarray(y_results).view(dt)
	_, cnt = np.unique(b, return_counts=True)

	return cnt


# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=True, fill_mode='mean')

# test deepmodel
deep = DeepKDModel()
deep.train(x_train, y_train)
deep_y_pred = deep.test(x_test, y_test)


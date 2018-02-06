# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# load_data.py

# Functions for loading data

# by Peter Lillian (hi peter)
# -------------------------------------------------------------------------------------------------

import pickle as pkl
import numpy as np
import numpy.ma as ma

# fill all cells containing NaN with the average value
# this is a basic way of dealing with missing data
def fill_nan(data):
	data = np.where(np.isnan(data), ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
	return data

# load the dataset from pkl file

# TODO: move this to root
def load(one_hot=True):
	f = open('../data/kd_dataset.pkl','rb')
	x_train, x_test, y_train, y_test = pkl.load(f)
	f.close()
	
	# one-hot encode y
	if (one_hot):
		y_train = np.eye(np.max(y_train) + 1)[y_train]
		y_test = np.eye(np.max(y_test) + 1)[y_test]

	return fill_nan(x_train), fill_nan(x_test), fill_nan(y_train), fill_nan(y_test)

# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# load_data.py

# Functions for loading data

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import pickle as pkl
import numpy as np

# fill all cells containing NaN with the average value
# this is a basic way of dealing with missing data
def fill_nan(data):
	print(data)
	column_avgs = np.nanmean(data, axis=0)
	inds = np.where(np.isnan(data))
	data[inds] = np.take(column_avgs, inds[1])
	print(data)
	return data

# load the dataset from pkl file
def load():
	f = open('data/kd_dataset.pkl','rb')
	x_train, x_test, y_train, y_test = pkl.load(f)
	f.close()
	return fill_nan(x_train), fill_nan(x_test), fill_nan(y_train), fill_nan(y_test)

# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# load_data.py

# Functions for loading data

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import pickle as pkl
import numpy as np
import numpy.ma as ma

# Fill NaN entries in data matrix (i.e. perform imputation)
	# Mode = 'mean': fill NaN entries using mean of that feature
	# Mode = 'knn': take K-nearest neighbors, calculate average over their feature values
def fill_nan(data, mode='mean', k=5):
	# Mean imputation
	if mode == 'mean':
		data = np.where(np.isnan(data), ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
	
	# KNN imputation - TODO: implement this without fancyimpute
	# elif mode == 'knn':
	# 	if np.isnan(data).any().any(): # If has NaN entries, then impute
	# 		data = KNN(k=k).complete(data)

	return data

# load the dataset from pkl file

# Load data from pickle dump
# Returns: x_train, x_test, y_train, y_test
	# one_hot: True = labels are one-hot vectors ([Not KD, KD]), False = labels are values (1 = KD)
	# fill_mode: how to fill NaN values (see fill_nan())
	# k: how many nearest neighbors to look at for KNN-based imputation
def load(one_hot=True, fill_mode='mean', k=5):
	try:
		f = open('../data/kd_dataset.pkl','rb')
	except:
		f = open('data/kd_dataset.pkl','rb')
	x_train, x_test, y_train, y_test = pkl.load(f)
	f.close()
	
	# one-hot encode y
	if (one_hot):
		y_train = np.eye(np.max(y_train) + 1)[y_train]
		y_test = np.eye(np.max(y_test) + 1)[y_test]

	return fill_nan(x_train, mode=fill_mode, k=k), fill_nan(x_test, mode=fill_mode, k=k), \
		fill_nan(y_train, mode=fill_mode, k=k), fill_nan(y_test, mode=fill_mode, k=k)

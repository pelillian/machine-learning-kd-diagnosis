# -------------------------------------------------------------------------------------------------
# MACHINE LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# load_data.py

# Functions for loading data

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import pickle as pkl

def load():
	f = open('data/kd_dataset.pkl','rb')
	x_train, x_test, y_train, y_test = pkl.load(f)
	f.close()
	return x_train, x_test, y_train, y_test

def get_train():
	x_train, x_test, y_train, y_test = load()
	return [x_train, y_train]

def get_test():
	x_train, x_test, y_train, y_test = load()
	return [x_test, y_test]
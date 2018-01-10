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
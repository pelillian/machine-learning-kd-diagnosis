# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# load_data.py

# Functions for loading data

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import pickle as pkl
import pandas as pd
import numpy as np
import numpy.ma as ma # masked array

# Standardize columns in a single dataframe (mean 0, variance 1) while ignoring NaNs
def standardize_df(df):
	# Convert each column to z-scores
	for col in list(df.columns):
		feature_mean = df[col].mean()
		feature_std = df[col].std(ddof=0)

		# Mean-centering
		df[col] = df[col] - feature_mean

		# Standardizing (divide by std-dev)
		if feature_std != 0:
			df[col] = df[col] / feature_std

# Standardize columns in train/test dataframes (mean 0, variance 1) while ignoring NaNs
	# Use df_train's (mean, std) for both dataframes for consistency
def standardize_dfs(df_train, df_test):
	assert set(df_train.columns) == set(df_test.columns)

	# Convert each column to z-scores
	for col in list(df_train.columns):
		feature_mean = df_train[col].mean()
		feature_std = df_train[col].std(ddof=0)

		# Mean-centering
		df_train[col] = df_train[col] - feature_mean
		df_test[col] = df_test[col] - feature_mean

		# Standardizing (divide by std-dev)
		if feature_std != 0:
			df_train[col] = df_train[col] / feature_std
			df_test[col] = df_test[col] / feature_std


# Fill NaN entries in data matrix (i.e. perform imputation)
	# Mode = 'none': don't fill in NaNs
	# Mode = 'mean': fill NaN entries using mean of that feature
	# Mode = 'knn': take K-nearest neighbors, calculate average over their feature values
def fill_nan(data, mode='mean', k=5):
	# "None" mode: Don't impute
	if mode == 'none':
		return data

	# Mean imputation
	if mode == 'mean':
		# Pandas DataFrame
		if isinstance(data, pd.DataFrame):
			data = data.fillna(data.mean())
		# Numpy array
		else:
			data = np.where(np.isnan(data), ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
	
	# KNN imputation
		# Documentation: https://github.com/iskandr/fancyimpute/blob/master/fancyimpute/knn.py
		# Assumes inputs are already standardized
	elif mode == 'knn':
		from fancyimpute import KNN

		# If has NaN entries, then impute
		if np.isnan(data).any().any():
			# Pandas DataFrame: get filled matrix, then create new DF
			if isinstance(data, pd.DataFrame):
				filled_array = KNN(k=k).complete(data.values)
				data = pd.DataFrame(data=filled_array, index=data.index, columns=data.columns)
			# Numpy array
			else:
				data = KNN(k=k).complete(data)

	return data


# Load dataset from pickle dump
# Returns: x_train, x_test, y_train, y_test
	# one_hot: True = labels are one-hot vectors ([Not KD, KD]), False = labels are values (1 = KD)
	# fill_mode: how to fill NaN values (see fill_nan())
	# k: how many nearest neighbors to look at for KNN-based imputation
def load(one_hot=False, fill_mode='mean', standardize=True, k=5, return_ids=True, return_pandas=False):
	# Load pickle dump
	try:
		f = open('../data/kd_dataset.pkl','rb')
	except:
		f = open('data/kd_dataset.pkl','rb')
	x_train, x_test, y_train, y_test = pkl.load(f)
	f.close()

	y_train = y_train[x_train['illday'] <= 10]
	y_test = y_test[x_test['illday'] <= 10]

	x_train = x_train[x_train['illday'] <= 10]
	x_test = x_test[x_test['illday'] <= 10]

	test_peternum = x_test['peternum']
	train_peternum = x_train['peternum']
	if not return_ids:
		del x_test['peternum']
		del x_train['peternum']

	# Standardize input data (mean 0, variance 1), while ignoring NaNs
	if fill_mode == 'knn': # Standardizing is required for kNN-based imputation
		standardize = True 
	if standardize:
		standardize_dfs(x_train, x_test)
	
	# One-hot encode y
	if (one_hot):
		y_train = np.eye(np.max(y_train) + 1)[y_train]
		y_test = np.eye(np.max(y_test) + 1)[y_test]

	if return_ids:
		x_test['peternum'] = test_peternum
		x_train['peternum'] = train_peternum
	
	# Return preprocessed (x_train, x_test, y_train, y_test)
	preprocessed_data = fill_nan(x_train, mode=fill_mode, k=k), fill_nan(x_test, mode=fill_mode, k=k), \
		fill_nan(y_train, mode=fill_mode, k=k), fill_nan(y_test, mode=fill_mode, k=k)

	if return_pandas == False:
		# Convert each element to numpy array
		for i in range(len(preprocessed_data)):
			preprocessed_data[i] = np.array(preprocessed_data[i])

	return preprocessed_data


# Load expanded dataset from pickle dump
# Returns: x_all, y_all, ids_all (numpy arrays)
def load_expanded(one_hot=False, fill_mode='mean', standardize=True, k=5, \
	return_ids=True, reduced_features=False, return_pandas=False):
	# Load pickle dump
	try:
		f = open('../data/kd_dataset_expanded.pkl','rb')
	except:
		f = open('./data/kd_dataset_expanded.pkl','rb')
	x_all, y_all, ids_all = pkl.load(f)
	f.close()

	# Reduced features: 18 features only (not 26)
	if reduced_features == True:
		reduced_features = ['illday', 'rash', 'redeyes', 'redplt', 'clnode', \
		'redhands', 'pwbc', 'ppolys', 'pbands', 'plymphs', 'pmonos',\
		'peos', 'pesr', 'pcrp', 'pplts', 'palt', 'pggt', 'zhemo']
		x_all = x_all[reduced_features] # update dataframe
	# Expanded features: one-hot ethnicity (0-9)
	else:
		ethnicity_mappings = {
			1: 'asian',
			2: 'black', 
			3: 'caucasian',
			4: 'hispanic',
			6: 'multiracial',
			7: 'american_indian',
			8: 'pacific_islander'
		}
		for ethnicity_val, ethnicity_name in ethnicity_mappings.items():
			x_all['eth_{}'.format(ethnicity_name)] = (x_all['eth'] == ethnicity_val).astype(int)

		x_all.drop('eth', axis=1, inplace=True)

	y_all = y_all[x_all['illday'] <= 10]
	ids_all = ids_all[x_all['illday'] <= 10]

	x_all = x_all[x_all['illday'] <= 10]

	# Standardize
	if fill_mode == 'knn':
		standardize = True
	if standardize:
		standardize_df(x_all)

	# One-hot encode y
	if (one_hot):
		y_all = np.eye(np.max(y_all) + 1)[y_all]

	# Fill NaNs
	x_filled = fill_nan(x_all, mode=fill_mode, k=k)

	if return_pandas == False:
		# Convert to numpy.ndarray
		x_filled = x_filled.values
		y_all = y_all.values
		ids_all = ids_all.values

	return (x_filled, y_all, ids_all)

# Load test dataset from pickle dump (18 features)
# Returns: x_all, ids_all (numpy arrays)
def load_test(fill_mode='mean', standardize=True, k=5, \
	return_ids=True, return_pandas=False):
	# Load pickle dump
	try:
		f = open('../data/kd_dataset_test.pkl','rb')
	except:
		f = open('./data/kd_dataset_test.pkl','rb')
	x_all, ids_all = pkl.load(f)
	f.close()
	
	ids_all = ids_all[x_all['illday'] <= 10]

	x_all = x_all[x_all['illday'] <= 10]

	# Standardize
	if fill_mode == 'knn':
		standardize = True
	if standardize:
		standardize_df(x_all)

	# Fill NaNs
	x_filled = fill_nan(x_all, mode=fill_mode, k=k)

	if return_pandas == False:
		# Convert to numpy.ndarray
		x_filled = x_filled.values
		ids_all = ids_all.values

	return (x_filled, ids_all)

# Load Jan 2019 multisite training dataset (1925 rows)
def load_multisite(one_hot=False, fill_mode='mean', standardize=True, k=5, \
		return_ids=True, return_pandas=False):
	# Load pickle dump
	try:
		f = open('../data/kd_dataset_multisite.pkl','rb')
	except:
		f = open('./data/kd_dataset_multisite.pkl','rb')
	x_all, y_all, ids_all = pkl.load(f)
	f.close()

	# Drop patients with illday > 10
	y_all = y_all[x_all['illday'] <= 10]
	ids_all = ids_all[x_all['illday'] <= 10]
	x_all = x_all[x_all['illday'] <= 10]

	# Standardize
	if fill_mode == 'knn':
		standardize = True
	if standardize:
		standardize_df(x_all)

	# One-hot encode y
	if (one_hot):
		y_all = np.eye(np.max(y_all) + 1)[y_all]

	# Fill NaNs
	x_filled = fill_nan(x_all, mode=fill_mode, k=k)

	if return_pandas == False:
		# Convert to numpy.ndarray
		x_filled = x_filled.values
		y_all = y_all.values
		ids_all = ids_all.values

	return (x_filled, y_all, ids_all)

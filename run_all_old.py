# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# from deepnet.deep_model import DeepKDModel
# from xgbst.xgboost_model import XGBoostKDModel

from preprocess import load_data

# Beta for fbeta_score
BETA = 1.5 # 0-1 favors precision, >1 (up to infinity) favors recall
CLASS_WEIGHT = "none" # set to "none" or "balanced"
USE_SMOTE = True

# ScikitModel wrapper class
class ScikitModel:
	def __init__(self, skmodel, params):
		self.skmodel = skmodel
		self.cv_scorer = make_scorer(fbeta_score, beta=BETA) # optimize for fbeta_score
		self.paramsearch = GridSearchCV(self.skmodel, params, cv=5, scoring=self.cv_scorer, verbose=True)

	def train_test(self, x_train, x_test, y_train, y_test):
		# params = self.skmodel.get_params(deep=True)
		# print(params)
		self.paramsearch.fit(x_train, y_train)
		y_pred = self.paramsearch.predict(x_test)
		return y_pred

# Explain TN, FP, FN, TP
def compute_stats(y_pred, y_test):
	if y_test.ndim > 1:
		y_results = np.column_stack((y_test[:, 1], y_pred))
	else:
		y_results = np.column_stack((y_test, y_pred))
	y_arr = np.dtype((np.void, y_results.dtype.itemsize * y_results.shape[1]))
	contigview = np.ascontiguousarray(y_results).view(y_arr)
	return np.unique(contigview, return_counts=True)[1].tolist()

# Explain TN, FP, FN, TP
def explain_stats(stats, model_name):
	fc_total = stats[0] + stats[1]
	kd_total = stats[2] + stats[3]
	filename = model_name + ".txt"
	# By default we append to the results file so we don't erase previous results
	with open(filename, "a") as resultsfile:
		fc_as_fc = (stats[0] / fc_total) * 100
		print("FC Classified as FC: " + str(stats[0]) + ", (" + str(fc_as_fc) + " %)", file=resultsfile)
		fc_as_kd = (stats[1] / fc_total) * 100
		print("FC Classified as KD: " + str(stats[1]) + ", (" + str(fc_as_kd) + " %)", file=resultsfile)
		kd_as_fc = (stats[2] / kd_total) * 100
		print("KD Classified as FC: " + str(stats[2]) + ", (" + str(kd_as_fc) + " %)", file=resultsfile)
		kd_as_kd = (stats[3] / kd_total) * 100
		print("KD Classified as KD: " + str(stats[3]) + ", (" + str(kd_as_kd) + " %)", file=resultsfile)
		print("", file=resultsfile)

# Train and evaluate model, print out results
def test_model(model, x, y, model_name, return_ids=True):
	print(model_name)

	stats_arr = []
	kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=90007)

	# this var will hold the pnum and pred for the entire set
	all_pnum_pred = []
	for train_idx, test_idx in kf.split(x, y):
		x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]
		if (USE_SMOTE):
			sm = SMOTE(random_state=hash("usc trojans") % (2**32 - 1))
			print(Counter(y_train))
			x_train, y_train = sm.fit_sample(x_train, y_train)
			print(Counter(y_train))
		# ids_train can't be used if SMOTE is run
		_, ids_test = ids[train_idx], ids[test_idx]

		# if return_ids:
		# 	pnum_x_test = x_test[:, 0]
		# 	x_train = np.delete(x_train, 0, axis=1)
		# 	x_test = np.delete(x_test, 0, axis=1)

		y_pred = model.train_test(x_train, x_test, y_train, y_test)
		stats_arr.append(compute_stats(y_pred, y_test))

		if return_ids == True:
			all_pnum_pred.append(np.column_stack((ids_test, y_pred)))

	if return_ids == True:
		all_pnum_pred = np.vstack(all_pnum_pred).astype(int)
		filename = model_name + ".out"
		np.savetxt(filename, all_pnum_pred, delimiter=',')

	explain_stats(np.mean(stats_arr, axis=0), model_name)

# Load expanded dataset
x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean')

# print("Our Models:")

# Test DNN Model
# test_model(DeepKDModel(), x, y, "Deep Model")

# Test XGBoost Model
# test_model(XGBoostKDModel(), x, y, "XGBoost Model")
# Our XGBoost class will not work with SMOTE because of how it loads the feature names

print("")
print("Scikit Models:")

# Logistic Regression
params = {
	# 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
	# 'multi_class': ['ovr', 'multinomial'],
	'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
	# 'penalty': ['l1', 'l2']
}
if (CLASS_WEIGHT != "none" and not USE_SMOTE):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
test_model(ScikitModel(LogisticRegression(), params), x, y, "Logistic Regression")

# SVM/SVC
params = {
	'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	'kernel': ['linear', 'rbf', 'poly']
}
if (CLASS_WEIGHT != "none" and not USE_SMOTE):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
test_model(ScikitModel(SVC(), params), x, y, "Support Vector Classification")

# Random Forest
params = {
	'max_features': ['auto', 'sqrt'],
	'n_estimators': [100, 200, 400, 800, 1600],
	'min_samples_leaf': [1, 2, 4],
	'min_samples_split': [2, 4, 8, 16],
	'bootstrap': [True, False],
	'max_depth': [10, 20, 40, 80, None]
}
if (CLASS_WEIGHT != "none" and not USE_SMOTE):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
test_model(ScikitModel(RandomForestClassifier(), params), x, y, "Random Forest")

# K-NN
params = {
	'n_neighbors':[1, 2, 3, 5, 9, 17],
	'leaf_size':[1,2,3,5],
	'weights':['uniform', 'distance'],
	'algorithm':['auto', 'ball_tree','kd_tree','brute'],
	'n_jobs':[-1]
}
if (CLASS_WEIGHT != "none" and not USE_SMOTE):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
test_model(ScikitModel(KNeighborsClassifier(4), params), x, y, "K Nearest Neighbors")


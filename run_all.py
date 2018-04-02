# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from deepnet.deep_model import DeepKDModel
from xgbst.xgboost_model import XGBoostKDModel

from preprocess import load_data

class ScikitModel:
	def __init__(self, skmodel, params, verbose=False):
		self.skmodel = skmodel
		self.paramsearch = GridSearchCV(self.skmodel, params, cv=5)
		self.verbose = verbose

	def train_test(self, x_train, x_test, y_train, y_test):
		params = self.skmodel.get_params(deep=True)
		print(params)
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
def explain_stats(stats):
	fc_total = stats[0] + stats[1]
	kd_total = stats[2] + stats[3]
	fc_as_fc = (stats[0] / fc_total) * 100
	print("FC Classified as FC: " + str(stats[0]) + ", (" + str(fc_as_fc) + " %)")
	fc_as_kd = (stats[1] / fc_total) * 100
	print("FC Classified as KD: " + str(stats[1]) + ", (" + str(fc_as_kd) + " %)")
	kd_as_fc = (stats[2] / kd_total) * 100
	print("KD Classified as FC: " + str(stats[2]) + ", (" + str(kd_as_fc) + " %)")
	kd_as_kd = (stats[3] / kd_total) * 100
	print("KD Classified as KD: " + str(stats[3]) + ", (" + str(kd_as_kd) + " %)")

# Train and evaluate model, print out results
def test_model(model, x, y):
	stats_arr = []
	kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=90007)
	for train_idx, test_idx in kf.split(x, y):
		x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]
		y_pred = model.train_test(x_train, x_test, y_train, y_test)
		stats_arr.append(compute_stats(y_pred, y_test))

	explain_stats(np.mean(stats_arr, axis=0))


# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=False, fill_mode='mean')
x, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

# print("Our Models:")
# print("Deep Model")
# test_model(DeepKDModel(), x, y)

# print("XGBoost Model")
# test_model(XGBoostKDModel(), x, y)

print("")
print("Scikit Models:")

print("Logistic Regression")
params = {
	'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
	# 'multi_class': ['ovr', 'multinomial'],
	'class_weight': [None, 'balanced'],
	'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]#,
	# 'penalty': ['l1', 'l2']
}
test_model(ScikitModel(LogisticRegression(), params), x, y)

print("Support Vector Classification")
params = {
	'C': np.logspace(-3, 10, 14),
	'gamma': np.logspace(-10, 4, 15),
	'kernel': ['linear', 'rbf', 'poly']
}
test_model(ScikitModel(SVC(), params), x, y)

print("Random Forest")
params = {
	'max_features': ['auto', 'sqrt'],
	'n_estimators': [100, 200, 400, 600, 800, 1200, 1600, 2000],
	'min_samples_leaf': [1, 2, 4],
	'min_samples_split': [2, 4, 8, 16],
	'bootstrap': [True, False],
	'max_depth': [10, 20, 30, 40, 50, 60, 80, 100, None],
}
test_model(ScikitModel(RandomForestClassifier(), params), x, y)

print("K Nearest Neighbors")
params = {
	'n_neighbors':[1, 2, 3, 5, 7, 9, 13, 17, 25],
	'leaf_size':[1,2,3,5],
	'weights':['uniform', 'distance'],
	'algorithm':['auto', 'ball_tree','kd_tree','brute'],
	'n_jobs':[-1]
	}
test_model(ScikitModel(KNeighborsClassifier(4), params), x, y)


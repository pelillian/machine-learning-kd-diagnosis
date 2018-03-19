# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from deepnet.deep_model import DeepKDModel
from xgbst.xgboost_model import XGBoostKDModel

from preprocess import load_data

class ScikitModel:
	def __init__(self, model, verbose=False):
		self.model = model
		self.verbose = verbose

	def train_test(self, x_train, x_test, y_train, y_test):
		self.model.fit(x_train, y_train)
		y_pred = self.model.predict(x_test)
		return y_pred

def compute_stats(y_pred, y_test):
	if y_test.ndim > 1:
		y_results = np.column_stack((y_test[:, 1], y_pred))
	else:
		y_results = np.column_stack((y_test, y_pred))
	y_arr = np.dtype((np.void, y_results.dtype.itemsize * y_results.shape[1]))
	contigview = np.ascontiguousarray(y_results).view(y_arr)
	return np.unique(contigview, return_counts=True)[1]

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

def test_model(model, x_train, x_test, y_train, y_test):
	y_pred = model.train_test(x_train, x_test, y_train, y_test)
	stats = compute_stats(y_pred, y_test)

	explain_stats(stats)


# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=True, fill_mode='mean')

print("Deep Model")
test_model(DeepKDModel(), x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test = load_data.load(one_hot=False, fill_mode='mean')

print("XGBoost Model")
test_model(XGBoostKDModel(), x_train, x_test, y_train, y_test)

print("Logistic Regression")
test_model(ScikitModel(LogisticRegression()), x_train, x_test, y_train, y_test)

print("Support Vector Classification")
test_model(ScikitModel(SVC()), x_train, x_test, y_train, y_test)

print("Random Forest")
test_model(ScikitModel(RandomForestClassifier()), x_train, x_test, y_train, y_test)
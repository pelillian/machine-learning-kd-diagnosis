# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


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
	# print(str(stats_arr).encode("utf-8").decode("ascii"))
	# print(np.mean(stats_arr, axis=0))
	explain_stats(np.mean(stats_arr, axis=0))


# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=False, fill_mode='mean')
x, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

print("Our Models:")
print("Deep Model")
test_model(DeepKDModel(), x, y)

print("XGBoost Model")
test_model(XGBoostKDModel(), x, y)

print("")
print("Scikit Models:")

print("Logistic Regression")
test_model(ScikitModel(LogisticRegression()), x, y)

print("Support Vector Classification: Linear Kernel")
test_model(ScikitModel(SVC(kernel='linear')), x, y)

print("Support Vector Classification: RBF Kernel")
test_model(ScikitModel(SVC(kernel='rbf')), x, y)

print("Support Vector Classification: Polynomial Kernel")
test_model(ScikitModel(SVC(kernel='poly')), x, y)

print("Gaussian Process Classifier")
test_model(ScikitModel(GaussianProcessClassifier(1.0 * RBF(1.0))), x, y)

print("Random Forest")
test_model(ScikitModel(RandomForestClassifier()), x, y)

print("AdaBoost Classifier")
test_model(ScikitModel(AdaBoostClassifier()), x, y)

print("Multi-layer Perceptron")
test_model(ScikitModel(MLPClassifier(max_iter=400)), x, y)

print("K Nearest Neighbors")
test_model(ScikitModel(KNeighborsClassifier(4)), x, y)

print("Gaussian Native Bayes")
test_model(ScikitModel(GaussianNB()), x, y)

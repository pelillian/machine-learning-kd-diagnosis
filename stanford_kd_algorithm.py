# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# stanford_kd_algorithm.py

# This script implements the stanford algorithm from A Classification Tool for Differentiation of
# Kawasaki Disease from Other Febrile Illnesses by Hao et al.

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from model_helpers.models import *
from preprocess import load_data

class StanfordModel:
	def __init__(self, rf_fc_threshold=0.4, rf_kd_threshold=0.6,
			rf_n_estimators=300, rf_max_features=1/3, verbose=True): # had to hardcode thresholds because they overlapped
		self.rf_fc_threshold = rf_fc_threshold
		self.rf_kd_threshold = rf_kd_threshold
		self.rf_n_estimators = rf_n_estimators
		self.rf_max_features = rf_max_features
		self.verbose = verbose

		if self.verbose:
			print("STANFORD KD ALGORITHM")
			print("ABOUT TO GET DESTROYED BY USC")
			print("#BEATtheFARM --- Fight On!")

	def train(self, x_train, y_train):
		### Linear Discriminant Analysis
		self.lda = LinearDiscriminantAnalysis()
		self.lda.fit(x_train, y_train)
		lda_train_proba = self.lda.predict_proba(x_train)[:, 1]

		self.lda_fc_threshold, self.lda_kd_threshold = get_fc_kd_thresholds(lda_train_proba, y_train)

		indeterminate_train = np.array(np.logical_and(lda_train_proba > self.lda_fc_threshold, lda_train_proba < self.lda_kd_threshold))

		x_indeterminate_train = x_train[indeterminate_train]
		y_indeterminate_train = y_train[indeterminate_train]

		### Random Forest ###
		self.rf = RandomForestClassifier(n_estimators=self.rf_n_estimators, max_features=self.rf_max_features)
		self.rf.fit(x_indeterminate_train, y_indeterminate_train)
		rf_train_proba = self.rf.predict_proba(x_indeterminate_train)[:, 1]

		final_proba = np.copy(lda_train_proba)
		final_proba[indeterminate_train] = rf_train_proba

		final_roc = roc_curve(y_train, final_proba)
		return auc(final_roc[0], final_roc[1])

	# Predict on x_test, return probability that each patient is KD
	def predict_proba(self, x_test):
		### Test ###
		lda_test_proba = self.lda.predict_proba(x_test)[:, 1]
		indeterminate_test = np.array(np.logical_and(lda_test_proba > self.lda_fc_threshold, lda_test_proba < self.lda_kd_threshold))

		x_indeterminate_test = x_test[indeterminate_test]
		y_indeterminate_test = y_test[indeterminate_test]

		rf_test_proba = self.rf.predict_proba(x_indeterminate_test)[:, 1]

		# lda_fc = np.array(lda_test_proba <= self.lda_fc_threshold)
		# lda_kd = np.array(lda_test_proba >= self.lda_kd_threshold)

		# rf_fc = np.array(rf_test_proba <= self.rf_fc_threshold)
		# rf_kd = np.array(rf_test_proba >= self.rf_kd_threshold)

		final_indeterminate_test = np.zeros(indeterminate_test.shape, dtype=bool)
		final_indeterminate_test[indeterminate_test] = np.array(np.logical_and(rf_test_proba > self.rf_fc_threshold, rf_test_proba < self.rf_kd_threshold))

		final_proba = np.copy(lda_test_proba)
		final_proba[indeterminate_test] = rf_test_proba

		return final_proba

	# Predict on x_test, return binary y_pred
	def predict(self, x_test, threshold=0.5):
		y_prob = self.predict_proba(x_test) # probability of KD
		y_pred = apply_threshold(y_prob, threshold)
		return y_pred

	# Train on x_train and y_train, and predict on x_test
	def train_test(self, x_train, x_test, y_train, y_test, threshold=0.5):
		self.train(x_train, y_train)
		return self.predict(x_test, threshold=threshold)


if __name__ == "__main__":
	# Load dataset (we reduce the features like they did in the stanford paper)
	x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean', reduced_features=True)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47252)

	# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	# TODO: test on this
	# x_test, ids_test = load_data.load_test(fill_mode='mean')

	stan = StanfordModel()
	stan.train(x_train, y_train)
	final_proba = stan.predict_proba(x_test)

	final_roc = roc_curve(y_test, final_proba)
	print('final roc', auc(final_roc[0], final_roc[1]))

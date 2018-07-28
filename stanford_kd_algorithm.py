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
		self.calibrated = False

		if self.verbose:
			print("STANFORD KD ALGORITHM")
			print("ABOUT TO GET DESTROYED BY USC")
			print("#BEATtheFARM --- Fight On!")

	# Train & Calibrate model (fit LDA and RF on half of train-set, perform calibration on other set)
		# Store thresholds in self.{lda/rf}_{kd/fc}_threshold
		# If refit=True, refit LDA and RF on entire train-set for inference
	def train_calibrate(self, x_train, y_train, calibration_set_size=0.2, random_state=90007, refit=False):
		# Split train and calibrate sets
		x_train_calibrate, x_calibrate, y_train_calibrate, y_calibrate = train_test_split(x_train, y_train, 
			test_size=calibration_set_size, random_state=random_state, stratify=y_train)

		# Calibrate LDA
		self.lda = LinearDiscriminantAnalysis()
		self.lda.fit(x_train_calibrate, y_train_calibrate)
		lda_calibrate_proba = self.lda.predict_proba(x_calibrate)[:, 1]
		self.lda_fc_threshold, self.lda_kd_threshold = get_fc_kd_thresholds(lda_calibrate_proba, y_calibrate)

		# Calibrate RF
		self.rf = RandomForestClassifier(n_estimators=self.rf_n_estimators, max_features=self.rf_max_features)
		self.rf.fit(x_train_calibrate, y_train_calibrate)
		rf_calibrate_proba = self.lda.predict_proba(x_calibrate)[:, 1]
		self.rf_fc_threshold, self.rf_kd_threshold = get_fc_kd_thresholds(rf_calibrate_proba, y_calibrate)

		# Refit on entire train-set after calibration
		if refit == True:
			self.lda.fit(x_train, y_train)
			self.rf.fit(x_train, y_train)

		self.calibrated = True

	# Train only (fit LDA and RF) -- don't calibrate
	def train(self, x_train, y_train):
		self.lda.fit(x_train, y_train)
		self.rf.fit(x_train, y_train)

	# Predict on x_test, return probability that each patient is KD
	def predict_proba(self, x_test):
		if self.calibrated == False:
			print('WARNING: called predict_proba on Stanford model without pre-calibrating!')

		### Test ###
		lda_test_proba = self.lda.predict_proba(x_test)[:, 1]
		indeterminate_test = np.array(np.logical_and(lda_test_proba > self.lda_fc_threshold, lda_test_proba < self.lda_kd_threshold))

		x_indeterminate_test = x_test[indeterminate_test]

		final_proba = np.copy(lda_test_proba)
		
		# If there are indeterminates, use RF to generate stage-2 predictions
		if x_indeterminate_test.shape[0] > 0:
			rf_test_proba = self.rf.predict_proba(x_indeterminate_test)[:, 1]
			final_indeterminate_test = np.zeros(indeterminate_test.shape, dtype=bool)
			final_indeterminate_test[indeterminate_test] = np.array(np.logical_and(rf_test_proba > self.rf_fc_threshold, rf_test_proba < self.rf_kd_threshold))
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

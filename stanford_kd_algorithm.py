# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# stanford_kd_algorithm.py

# This script implements the stanford algorithm from A Classification Tool for Differentiation of
# Kawasaki Disease from Other Febrile Illnesses by Hao et al.

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from model_helpers.models import *
from preprocess import load_data

# Load dataset (we reduce the features like they did in the stanford paper)
x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean', reduced_features=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47252)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# TODO: test on this
# x_test, ids_test = load_data.load_test(fill_mode='mean')

print("STANFORD KD ALGORITHM")
print("#BEATtheFARM --- Fight On!")

### Linear Discriminant Analysis
lda = ScikitModel(LinearDiscriminantAnalysis(),
	params={},
	random_search=False,
	scoring='roc_auc',
	verbose=True)
lda.train(x_train, y_train)
lda_train_prob = lda.predict_proba(x_train)
lda_train_pred = lda.predict(x_train, threshold=0.5)
lda_test_prob = lda.predict_proba(x_test)
lda_test_pred = lda.predict(x_test, threshold=0.5)

fc_threshold, kd_threshold = get_fc_kd_thresholds(lda_test_prob, y_test)

indeterminate_train = np.array(np.logical_and(lda_train_prob > fc_threshold, lda_train_prob < kd_threshold))
indeterminate_test = np.array(np.logical_and(lda_test_prob > fc_threshold, lda_test_prob < kd_threshold))

x_indeterminate_train = x_train[indeterminate_train]
x_indeterminate_test = x_test[indeterminate_test]
y_indeterminate_train = y_train[indeterminate_train]
y_indeterminate_test = y_test[indeterminate_test]

### Random Forest ###
rf_params = {
	'n_estimators': 300,
	'max_features': 1/3
}
if (CLASS_WEIGHT != "none"):
	rf_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
rf = ScikitModel(RandomForestClassifier(), 
	params=rf_params,
	random_search=True,
	n_iter=25,
	scoring='roc_auc',
	verbose=True)
rf.train(x_indeterminate_train, y_indeterminate_train)
rf_train_prob = rf.predict_proba(x_indeterminate_train)
rf_train_pred = rf.predict(x_indeterminate_train, threshold=0.5)
rf_test_prob = rf.predict_proba(x_indeterminate_test)
rf_test_pred = rf.predict(x_indeterminate_test, threshold=0.5)
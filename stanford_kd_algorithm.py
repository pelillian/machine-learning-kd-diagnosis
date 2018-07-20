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

# Load dataset (we reduce the features like they did in the stanford paper)
x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean', reduced_features=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47252)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# TODO: test on this
# x_test, ids_test = load_data.load_test(fill_mode='mean')

print("STANFORD KD ALGORITHM")
print("#BEATtheFARM --- Fight On!")

### Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
lda_train_proba = lda.predict_proba(x_train)[:, 1]
lda_test_proba = lda.predict_proba(x_test)[:, 1]
lda_test_pred = lda.predict(x_test)

lda_fc_threshold, lda_kd_threshold = get_fc_kd_thresholds(lda_test_pred, y_test)

indeterminate_train = np.array(np.logical_and(lda_train_proba > lda_fc_threshold, lda_train_proba < lda_kd_threshold))
indeterminate_test = np.array(np.logical_and(lda_test_proba > lda_fc_threshold, lda_test_proba < lda_kd_threshold))

x_indeterminate_train = x_train[indeterminate_train]
y_indeterminate_train = y_train[indeterminate_train]
x_indeterminate_test = x_test[indeterminate_test]

### Random Forest ###
rf = RandomForestClassifier(n_estimators=300, max_features=1/3)
rf.fit(x_indeterminate_train, y_indeterminate_train)
rf_train_proba = rf.predict_proba(x_indeterminate_train)[:, 1]
rf_test_pred = rf.predict(x_indeterminate_test)

rf_fc_threshold, rf_kd_threshold = get_fc_kd_thresholds(rf_train_pred, y_indeterminate_train)

### Test ###

rf_test_proba = rf.predict_proba(x_indeterminate_test)[:, 1]

lda_fc = np.array(lda_test_proba <= lda_fc_threshold)
lda_kd = np.array(lda_test_proba >= lda_kd_threshold)

rf_fc = np.array(rf_test_proba <= rf_fc_threshold)
rf_kd = np.array(rf_test_proba >= rf_kd_threshold)

final_indeterminate_test = np.array(np.logical_and(rf_test_proba > rf_fc_threshold, rf_test_proba < rf_kd_threshold))

print(lda_kd.shape, lda_fc.shape, rf_kd.shape, rf_fc.shape, final_indeterminate_test.shape)
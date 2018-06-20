# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# Peter Lillian
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
from model_helpers.models import *

from preprocess import load_data

# Beta for fbeta_score
BETA = 1.5 # 0-1 favors precision, >1 (up to infinity) favors recall
CLASS_WEIGHT = "none" # set to "none" or "balanced"
USE_SMOTE = True

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
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Logistic Regression")
test_model(ScikitModel(LogisticRegression(), params), x, y)

# SVM/SVC
params = {
	'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	'kernel': ['linear', 'rbf', 'poly']
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Support Vector Classification")
test_model(ScikitModel(SVC(), params), x, y)

# Random Forest
params = {
	'max_features': ['auto', 'sqrt'],
	'n_estimators': [100, 400, 1600],
	'min_samples_leaf': [1, 2, 4],
	'min_samples_split': [2, 6, 16],
	'bootstrap': [True, False],
	'max_depth': [10, 30, 80, None]
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Random Forest")
test_model(ScikitModel(RandomForestClassifier(), params), x, y)

# K-NN
params = {
	'n_neighbors':[1, 2, 3, 5, 9, 17],
	'leaf_size':[1,2,3,5],
	'weights':['uniform', 'distance'],
	'algorithm':['auto', 'ball_tree','kd_tree','brute'],
	'n_jobs':[-1]
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("K Nearest Neighbors")
test_model(ScikitModel(KNeighborsClassifier(4), params), x, y)


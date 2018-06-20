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
   'C': np.logspace(-2, 2, 5)
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Logistic Regression")
test_model(ScikitModel(LogisticRegression(), 
               params=params,
               random_search=False,
               scoring='roc_auc',
               verbose=True),
       x, y,
       allow_indeterminates=True)

# SVM/SVC
params = {
   'C': np.logspace(-3, 3, 100),
   'gamma': np.logspace(-3, 3, 100),
   'kernel': ['linear', 'rbf', 'poly']
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Support Vector Classification")
test_model(ScikitModel(SVC(probability=True),
			params=params,
			random_search=True,
			n_iter=100,
			scoring='roc_auc',
			verbose=True),
		x, y,
		allow_indeterminates=True)

# Random Forest
params = {
   'n_estimators': randint(10, 500),
   'max_features': randint(3, 15),
   'min_samples_split': randint(2, 50),
   'min_samples_leaf': randint(1, 50)
}
if (CLASS_WEIGHT != "none"):
	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Random Forest")
test_model(ScikitModel(RandomForestClassifier(), 
                       params=params,
                       random_search=True,
                       n_iter=250,
                       scoring='roc_auc',
                       verbose=True),
           x, y,
           allow_indeterminates=True)

# K-NN
# params = {
# 	'n_neighbors':[1, 2, 3, 5, 9, 17],
# 	'leaf_size':[1,2,3,5],
# 	'weights':['uniform', 'distance'],
# 	'algorithm':['auto', 'ball_tree','kd_tree','brute'],
# 	'n_jobs':[-1]
# }
# if (CLASS_WEIGHT != "none"):
# 	params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
# print("K Nearest Neighbors")
# test_model(ScikitModel(KNeighborsClassifier(4), params), x, y)

# Ensemble
clf1 = svm.SVC(probability=True)
clf2 = linear_model.LogisticRegression()

eclf = ensemble.VotingClassifier(
    estimators=[('svm', clf1), ('lr', clf2)],
    voting='soft')

params = {
    'svm__C': np.logspace(-3, 2, 100),
    'svm__gamma': np.logspace(-3, 2, 100),
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'lr__C': np.logspace(-3, 2, 100)
}

# Test model! 5-fold CV with hyperparameter optimization
clf = 

test_model(ScikitModel(
				eclf,
				params,
				random_search=True, 
				n_iter=100, 
				verbose=True),
		x, y, allow_indeterminates=True)
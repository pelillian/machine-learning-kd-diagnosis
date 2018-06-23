# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import numpy as np

from collections import Counter

from scipy.stats import randint

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

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
# print("")

print("Scikit Models:")
print("")

# Logistic Regression
lr_params = {
	'C': np.logspace(-2, 2, 5)
}
if (CLASS_WEIGHT != "none"):
	lr_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Logistic Regression")
test_model(ScikitModel(LogisticRegression(), 
				params=lr_params,
				random_search=False,
				scoring='roc_auc',
				verbose=True),
		x, y,
		allow_indeterminates=True
)
print()


# SVM/SVC
svc_params = {
	'C': np.logspace(-3, 2, 100),
	'gamma': np.logspace(-3, 2, 100),
	'kernel': ['linear', 'rbf', 'poly']
}
if (CLASS_WEIGHT != "none"):
	svc_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
print("Support Vector Classification")
test_model(ScikitModel(SVC(probability=True),
			params=svc_params,
			random_search=True,
			n_iter=25,
			scoring='roc_auc',
			verbose=True),
		x, y,
		allow_indeterminates=True)
print()



# Random Forest
# rf_params = {
#    'n_estimators': randint(50, 500),
#    'max_features': randint(3, 10),
#    'min_samples_split': randint(2, 50),
#    'min_samples_leaf': randint(1, 40)
# }
# if (CLASS_WEIGHT != "none"):
# 	rf_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
# print("Random Forest")
# test_model(ScikitModel(RandomForestClassifier(), 
# 				params=rf_params,
# 				random_search=True,
# 				n_iter=25,
# 				scoring='roc_auc',
# 				verbose=True),
# 		x, y,
# 		allow_indeterminates=True
# )
# print()


# # K-NN
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
# print("")



# Bagging Logistic Regression
bag_lr_params = {
	'base_estimator__C':np.logspace(-2, 2, 5),
	'n_estimators':randint(5, 50),
	'max_samples':np.logspace(-0.9, 0, 100),
	'max_features':randint(10, x.shape[1]),
	'bootstrap':[True, False],
	'bootstrap_features':[True, False]
}
bagging_lr = BaggingClassifier(
	base_estimator=LogisticRegression(),
	n_jobs=-1
)
print("Logistic Regression Bagging")
test_model(ScikitModel(
		bagging_lr,
		params=bag_lr_params,
		random_search=True,
		n_iter=25,
		verbose=1),
	x, y,
	allow_indeterminates=True
)
print()


# Bagging SVC
bag_svm_params = {
	'base_estimator__C': np.logspace(-3, 2, 100),
	'base_estimator__gamma': np.logspace(-3, 2, 100),
	'base_estimator__kernel': ['linear', 'rbf', 'poly'],
	'base_estimator__probability': [True, False],
	'n_estimators': randint(5, 50),
	"max_samples": np.logspace(-0.9, 0, 100),
	"max_features": randint(10, x.shape[1]),
	"bootstrap": [True, False],
	"bootstrap_features": [True, False]
}
bagging_svc = BaggingClassifier(
	base_estimator=SVC(),
	n_jobs=-1
)
print("SVC Bagging")
test_model(ScikitModel(
				bagging_svc,
				params=bag_svm_params,
				random_search=True,
				n_iter=25,
				verbose=1),
		x, y,
		allow_indeterminates=True
)
print()


# Voting Ensemble
clf1 = SVC(probability=True)
clf2 = LogisticRegression()
# clf3 = RandomForestClassifier()

eclf = VotingClassifier(
    estimators=[
    	('svm', clf1), 
    	('lr', clf2)
    	# ('rf', clf3)
    ],
    voting='soft',
    n_jobs=-1
)
eclf_params = {
    'svm__C': np.logspace(-3, 2, 100),
	'svm__gamma': np.logspace(-3, 2, 100),
	'svm__kernel': ['rbf', 'poly'],
    'lr__C': np.logspace(-3, 2, 100)
	# 'rf__n_estimators': randint(50, 500),
	# 'rf__max_features': randint(3, 10),
	# 'rf__min_samples_split': randint(2, 50),
	# 'rf__min_samples_leaf': randint(1, 40)
}
print("Voting Ensemble")
test_model(ScikitModel(
				eclf,
				eclf_params,
				random_search=True, 
				n_iter=25,
				verbose=True),
		x, y,
		allow_indeterminates=True
)
print()
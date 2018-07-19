# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# predict_test_set.py

# Trains on complete train-set (18 features), predicts on test-set

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import randint

from collections import Counter
import json
import csv

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
import xgboost as xgb

# from deepnet.deep_model import DeepKDModel
# from xgbst.xgboost_model import XGBoostKDModel
from model_helpers.models import *

from preprocess import load_data


# Load data: 18 features
x_train, y_train, ids_train = load_data.load_expanded(one_hot=False, fill_mode='mean', reduced_features=True)
x_test, ids_test = load_data.load_test(fill_mode='mean')


### FOR EACH MODEL: ###
	# Fit GridSearchCV on entire train-set
	# Predict on entire test-set: return probability? or KD/FC/Indeterminate?
	# Write predictions to .csv file: (model_name, id_test, predicted probability (0-1))

RANDOM_STATE = 0
ALLOW_INDETERMINATES = True
N_JOBS = 1

with open('./data/test_predictions/test_preds.csv', 'w') as f:

	writer = csv.DictWriter(f, fieldnames=['model', 'patient_id', 'kd_probability'])
	writer.writeheader()

	### LOGISTIC REGRESSION ###
	print('LOGISTIC REGRESSION')
	lr_params = {
		'C': np.logspace(-2, 2, 5)
	}

	lr = ScikitModel(LogisticRegression(), 
					params=lr_params,
					random_search=False,
					scoring='roc_auc',
					verbose=True)
	lr.train(x_train, y_train)
	lr_preds = lr.predict_proba(x_test)

	for i, probability in enumerate(lr_preds):
		writer.writerow({'model':'logistic_regression', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()


	### SUPPORT VECTOR CLASSIFIER ###
	print('SUPPORT VECTOR CLASSIFIER')
	svc_params = {
		'C': np.logspace(-3, 2, 100),
		'gamma': np.logspace(-3, 2, 100),
		'kernel': ['linear', 'rbf', 'poly']
	}

	svc = ScikitModel(SVC(probability=True),
					params=svc_params,
					random_search=True,
					n_iter=25,
					scoring='roc_auc',
					verbose=True)
	svc.train(x_train, y_train)
	svc_preds = svc.predict_proba(x_test)

	for i, probability in enumerate(svc_preds):
		writer.writerow({'model':'support_vector_classifier', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()

	### XGBOOST ###
	print('XGBOOST')
	xgb_params = {
		'n_estimators': randint(50, 500),
		'max_depth': randint(3, 10),
		'learning_rate': np.logspace(-2, 0, 100),
		'min_child_weight': randint(1, 5),
		'subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
		'colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	}

	xgb = ScikitModel(xgb.XGBClassifier(
						n_jobs=N_JOBS
					),
					params=xgb_params,
					random_search=True,
					n_iter=25,
					scoring='roc_auc',
					verbose=True)
	xgb.train(x_train, y_train)
	xgb_preds = xgb.predict_proba(x_test)

	for i, probability in enumerate(xgb_preds):
		writer.writerow({'model':'xgboost', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()


	### BAGGING LOGISTIC REGRESSION ###
	print('LOGISTIC REGRESSION BAGGING')
	bag_lr_params = {
		'base_estimator__C':np.logspace(-2, 2, 5),
		'n_estimators':randint(5, 50),
		'max_samples':np.logspace(-0.9, 0, 100),
		'max_features':randint(10, x_train.shape[1])
	}

	bagging_lr = ScikitModel(BaggingClassifier(
						base_estimator=LogisticRegression(),
						bootstrap=True,
						bootstrap_features=False,
						n_jobs=N_JOBS
					),
					params=bag_lr_params,
					random_search=True,
					n_iter=25,
					verbose=1)
	bagging_lr.train(x_train, y_train)
	bagging_lr_preds = bagging_lr.predict_proba(x_test)

	for i, probability in enumerate(bagging_lr_preds):
		writer.writerow({'model':'logistic_regression_ensemble', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()


	### BAGGING SVC ###
	bag_svm_params = {
		'base_estimator__C': np.logspace(-3, 2, 100),
		'base_estimator__gamma': np.logspace(-3, 2, 100),
		'base_estimator__kernel': ['linear', 'rbf', 'poly'],
		'base_estimator__probability': [True, False],
		'n_estimators': randint(5, 50),
		"max_samples": np.logspace(-0.9, 0, 100),
		"max_features": randint(10, x_train.shape[1])
	}

	bagging_svc = ScikitModel(BaggingClassifier(
						base_estimator=SVC(),
						bootstrap=True,
						bootstrap_features=False,
						n_jobs=N_JOBS
					),
					params=bag_svm_params,
					random_search=True,
					n_iter=25,
					verbose=1)
	bagging_svc.train(x_train, y_train)
	bagging_svc_preds = bagging_svc.predict_proba(x_test)

	for i, probability in enumerate(baggin_svc_preds):
		writer.writerow({'model':'support_vector_classifier_ensemble', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()



	### VOTING ENSEMBLE ###
	print('VOTING ENSEMBLE')
	clf1 = SVC(probability=True)
	clf2 = LogisticRegression()
	clf3 = xgb.XGBClassifier(n_jobs=N_JOBS)

	eclf = VotingClassifier(
	    estimators=[
	    	('svm', clf1), 
	    	('lr', clf2),
	    	('xgb', clf3)
	    ],
	    voting='soft',
	    n_jobs=N_JOBS
	)
	eclf_params = {
	    'svm__C': np.logspace(-3, 2, 100),
		'svm__gamma': np.logspace(-3, 2, 100),
		'svm__kernel': ['rbf', 'poly'],
	    'lr__C': np.logspace(-3, 2, 100),
		'xgb__n_estimators': randint(50, 500),
		'xgb__max_depth': randint(3, 10),
		'xgb__learning_rate': np.logspace(-2, 0, 100),
		'xgb__min_child_weight': randint(1, 5),
		'xgb__subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
		'xgb__colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	}

	eclf = ScikitModel(
					eclf,
					eclf_params,
					random_search=True, 
					n_iter=25,
					verbose=True)
	eclf.train(x_train, y_train)
	eclf_preds = eclf.predict_proba(x_test)

	for i, probability in enumerate(eclf_preds):
		writer.writerow({'model':'voting_ensemble', 'patient_id': ids_test[i], 'kd_probability': probability})

	print()



	### TODO: PREVIOUS PAPER'S LDA --> RF MODEL ###





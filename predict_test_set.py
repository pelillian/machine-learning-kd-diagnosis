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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

### MODELS TO USE: ###
	# Voting Ensemble, with indeterminates
	# Voting Ensemble, no indeterminates
	# LDA --> Voting Ensemble, with indeterminates
	# LDA --> Voting Ensemble, no indeterminates

RANDOM_STATE = 0
ALLOW_INDETERMINATES = True
N_JOBS = 1

np.random.seed(RANDOM_STATE)

with open('./data/test_predictions/test_preds.csv', 'w') as f:

	# writer = csv.DictWriter(f, fieldnames=['patient_id', 
	# 	'logistic_regression_kd_probability', 'support_vector_classifier_kd_probability', 'xgboost_kd_probability', 
	# 	'lr_bagging_kd_probability', 'svc_bagging_kd_probability', 'voting_ensemble_kd_probability'])

	writer = csv.DictWriter(f, fieldnames=[
		'patient_id',
		'voting_ensemble_with_indeterminates',
		'voting_ensemble_no_indeterminates',
		'2stage_lda_voting_ensemble_with_indeterminates',
		'2stage_lda_voting_ensemble_no_indeterminates'
		])

	writer.writeheader()

	### LOGISTIC REGRESSION ###
	# print('LOGISTIC REGRESSION')
	# lr_params = {
	# 	'C': np.logspace(-2, 2, 5)
	# }

	# lr = ScikitModel(LogisticRegression(), 
	# 				params=lr_params,
	# 				random_search=False,
	# 				scoring='roc_auc',
	# 				verbose=True)
	# lr.train(x_train, y_train)
	# lr_preds = lr.predict_proba(x_test)

	# print()


	### SUPPORT VECTOR CLASSIFIER ###
	# print('SUPPORT VECTOR CLASSIFIER')
	# svc_params = {
	# 	'C': np.logspace(-3, 2, 100),
	# 	'gamma': np.logspace(-3, 2, 100),
	# 	'kernel': ['linear', 'rbf', 'poly']
	# }

	# svc = ScikitModel(SVC(probability=True),
	# 				params=svc_params,
	# 				random_search=True,
	# 				n_iter=25,
	# 				scoring='roc_auc',
	# 				verbose=True)
	# svc.train(x_train, y_train)
	# svc_preds = svc.predict_proba(x_test)

	# print()

	### XGBOOST ###
	# print('XGBOOST')
	# xgb_params = {
	# 	'n_estimators': randint(50, 500),
	# 	'max_depth': randint(3, 10),
	# 	'learning_rate': np.logspace(-2, 0, 100),
	# 	'min_child_weight': randint(1, 5),
	# 	'subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
	# 	'colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	# }

	# xgboost = ScikitModel(xgb.XGBClassifier(
	# 					n_jobs=N_JOBS
	# 				),
	# 				params=xgb_params,
	# 				random_search=True,
	# 				n_iter=25,
	# 				scoring='roc_auc',
	# 				verbose=True)
	# xgboost.train(x_train, y_train)
	# xgboost_preds = xgboost.predict_proba(x_test)

	# print()


	### BAGGING LOGISTIC REGRESSION ###
	# print('LOGISTIC REGRESSION BAGGING')
	# bag_lr_params = {
	# 	'base_estimator__C':np.logspace(-2, 2, 5),
	# 	'n_estimators':randint(5, 50),
	# 	'max_samples':np.logspace(-0.9, 0, 100),
	# 	'max_features':randint(10, x_train.shape[1])
	# }

	# bagging_lr = ScikitModel(BaggingClassifier(
	# 					base_estimator=LogisticRegression(),
	# 					bootstrap=True,
	# 					bootstrap_features=False,
	# 					n_jobs=N_JOBS
	# 				),
	# 				params=bag_lr_params,
	# 				random_search=True,
	# 				n_iter=25,
	# 				verbose=1)
	# bagging_lr.train(x_train, y_train)
	# bagging_lr_preds = bagging_lr.predict_proba(x_test)

	# print()


	### BAGGING SVC ###
	# print('SVG BAGGING')
	# bag_svm_params = {
	# 	'base_estimator__C': np.logspace(-3, 2, 100),
	# 	'base_estimator__gamma': np.logspace(-3, 2, 100),
	# 	'base_estimator__kernel': ['linear', 'rbf', 'poly'],
	# 	'base_estimator__probability': [True, False],
	# 	'n_estimators': randint(5, 50),
	# 	"max_samples": np.logspace(-0.9, 0, 100),
	# 	"max_features": randint(10, x_train.shape[1])
	# }

	# bagging_svc = ScikitModel(BaggingClassifier(
	# 					base_estimator=SVC(probability=True),
	# 					bootstrap=True,
	# 					bootstrap_features=False,
	# 					n_jobs=N_JOBS
	# 				),
	# 				params=bag_svm_params,
	# 				random_search=True,
	# 				n_iter=25,
	# 				verbose=1)
	# bagging_svc.train(x_train, y_train)
	# bagging_svc_preds = bagging_svc.predict_proba(x_test)

	# print()



	### VOTING ENSEMBLE ###
	print('VOTING ENSEMBLE')
	clf1 = SVC(probability=True)
	clf2 = LogisticRegression()
	clf3 = xgb.XGBClassifier(n_jobs=N_JOBS)

	voting = VotingClassifier(
	    estimators=[
	    	('svm', clf1), 
	    	('lr', clf2),
	    	('xgb', clf3)
	    ],
	    voting='soft',
	    n_jobs=N_JOBS
	)
	voting_params = {
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

	voting = ScikitModel(
					voting,
					voting_params,
					random_search=True, 
					n_iter=25,
					verbose=True)

	# Train on whole train-set, predict with 0.5 threshold
	voting.train(x_train, y_train)
	voting_preds_no_indeterminates = voting.predict(x_test)

	# Train on 50%, calibrate on 50%, predict with calibrated thresholds
	voting.train_calibrate(x_train, y_train)
	voting_preds_with_indeterminates = voting.predict_calibrated(x_test)


	print()


	### 2 STAGE: LDA --> VOTING ENSEMBLE ###
	# Stage 1: LDA
	stage1 = LinearDiscriminantAnalysis()
	
	# Stage 2: LR-SVC-XGB Voting Classifier
	clf1 = SVC(probability=True)
	clf2 = LogisticRegression()
	clf3 = xgb.XGBClassifier(n_jobs=N_JOBS)
	voting = VotingClassifier(
	    estimators=[
	    	('svm', clf1), 
	    	('lr', clf2),
	    	('xgb', clf3)
	    ],
	    voting='soft',
	    n_jobs=N_JOBS
	)
	voting_params = {
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
	stage2 = RandomizedSearchCV(voting, voting_params, cv=5, n_iter=25, scoring='roc_auc', verbose=1, n_jobs=1)

	lda_voting = TwoStageModel(stage1, stage2, verbose=True)

	# Train on 50%, calibrate on 50%
	lda_voting.train_calibrate(x_train, y_train)

	# Final predictions
	lda_voting_preds_no_indeterminates = lda_voting.predict_calibrated(x_test, allow_indeterminates=False)
	lda_voting_preds_with_indeterminates = lda_voting.predict_calibrated(x_test, allow_indeterminates=True)




	print('Writing final predictions...')

	### WRITE PREDICTIONS ###
	for i, patient_id in enumerate(ids_test):
		writer.writerow({
			'patient_id': patient_id, 
			'voting_ensemble_no_indeterminates': voting_preds_no_indeterminates[i],
			'voting_ensemble_with_indeterminates': voting_preds_with_indeterminates[i],
			'2stage_lda_voting_ensemble_no_indeterminates': lda_voting_preds_no_indeterminates[i],
			'2stage_lda_voting_ensemble_with_indeterminates': lda_voting_preds_with_indeterminates[i],
			})

	print('Done!')
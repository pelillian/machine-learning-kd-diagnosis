# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# Peter Lillian & Lucas Hu
# -------------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import randint

from collections import Counter
import json

from stanford_kd_algorithm import StanfordModel

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

# Beta for fbeta_score
BETA = 1.5 # 0-1 favors precision, >1 (up to infinity) favors recall
CLASS_WEIGHT = "none" # set to "none" or "balanced"
USE_SMOTE = False
RANDOM_STATES = [90007, 0, 2018, 525, 7, 10, 777, 16, 99, 2048]
N_JOBS = 1
ALLOW_INDETERMINATES = True

# Load expanded dataset
x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean')

rocaucs_dict = {}
confusions_dict = {}

### RUN EXPERIMENTS ###
for random_state in RANDOM_STATES:

	print("------- RANDOM STATE: {} -------".format(random_state))
	print("")

	#### Stanford Algorithm ####
	avg_rocauc, confusions = test_model(StanfordModel(
					verbose=True),
				x, y,
				allow_indeterminates=True,
				random_state=random_state)

	if 'stanford' not in rocaucs_dict:
		rocaucs_dict['stanford'] = []
	rocaucs_dict['stanford'].append(avg_rocauc)

	if 'stanford' not in confusions_dict:
		confusions_dict['stanford'] = []
	confusions_dict['stanford'].append(confusions)

	print()

	#### Logistic Regression ###
	lr_params = {
		'C': np.logspace(-2, 2, 5)
	}
	if (CLASS_WEIGHT != "none"):
		lr_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
	print("LOGISTIC REGRESSION")
	avg_rocauc, confusions = test_model(ScikitModel(LogisticRegression(), 
					params=lr_params,
					random_search=False,
					scoring='roc_auc',
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'lr' not in rocaucs_dict:
		rocaucs_dict['lr'] = []
	rocaucs_dict['lr'].append(avg_rocauc)

	if 'lr' not in confusions_dict:
		confusions_dict['lr'] = []
	confusions_dict['lr'].append(confusions)

	print()


	### SVM/SVC ###
	svc_params = {
		'C': np.logspace(-3, 2, 100),
		'gamma': np.logspace(-3, 2, 100),
		'kernel': ['linear', 'rbf', 'poly']
	}
	if (CLASS_WEIGHT != "none"):
		svc_params['class_weight'] = CLASS_WEIGHT # how much to weigh FC patients over KD
	print("SUPPORT VECTOR CLASSIFIER")
	avg_rocauc, confusions = test_model(ScikitModel(SVC(probability=True),
					params=svc_params,
					random_search=True,
					n_iter=25,
					scoring='roc_auc',
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'svc' not in rocaucs_dict:
		rocaucs_dict['svc'] = []
	rocaucs_dict['svc'].append(avg_rocauc)

	if 'svc' not in confusions_dict:
		confusions_dict['svc'] = []
	confusions_dict['svc'].append(confusions)

	print()


	# ### Random Forest ###
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
	# 		allow_indeterminates=ALLOW_INDETERMINATES
	# )
	# print()


	#### XGBoost ###
	xgb_params = {
		'n_estimators': randint(50, 500),
		'max_depth': randint(3, 10),
		'learning_rate': np.logspace(-2, 0, 100),
		'min_child_weight': randint(1, 5),
		'subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
		'colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	}
	print('XGBOOST')
	avg_rocauc, confusions = test_model(ScikitModel(
					xgb.XGBClassifier(
						n_jobs=N_JOBS
					),
					params=xgb_params,
					random_search=True,
					n_iter=25,
					scoring='roc_auc',
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'xgb' not in rocaucs_dict:
		rocaucs_dict['xgb'] = []
	rocaucs_dict['xgb'].append(avg_rocauc)

	if 'xgb' not in confusions_dict:
		confusions_dict['xgb'] = []
	confusions_dict['xgb'].append(confusions)

	print()


	# ### K-NN ###
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


	### Bagging Logistic Regression ###
	bag_lr_params = {
		'base_estimator__C':np.logspace(-2, 2, 5),
		'n_estimators':randint(5, 50),
		'max_samples':np.logspace(-0.9, 0, 100),
		'max_features':randint(10, x.shape[1])
	}
	bagging_lr = BaggingClassifier(
		base_estimator=LogisticRegression(),
		bootstrap=True,
		bootstrap_features=False,
		n_jobs=N_JOBS
	)
	print("LOGISTIC REGRESSION BAGGING")
	avg_rocauc, confusions = test_model(ScikitModel(
					bagging_lr,
					params=bag_lr_params,
					random_search=True,
					n_iter=25,
					verbose=1),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'lr_bag' not in rocaucs_dict:
		rocaucs_dict['lr_bag'] = []
	rocaucs_dict['lr_bag'].append(avg_rocauc)

	if 'lr_bag' not in confusions_dict:
		confusions_dict['lr_bag'] = []
	confusions_dict['lr_bag'].append(confusions)

	print()


	### Bagging SVC ###
	bag_svm_params = {
		'base_estimator__C': np.logspace(-3, 2, 100),
		'base_estimator__gamma': np.logspace(-3, 2, 100),
		'base_estimator__kernel': ['linear', 'rbf', 'poly'],
		'base_estimator__probability': [True, False],
		'n_estimators': randint(5, 50),
		"max_samples": np.logspace(-0.9, 0, 100),
		"max_features": randint(10, x.shape[1])
	}
	bagging_svc = BaggingClassifier(
		base_estimator=SVC(),
		bootstrap=True,
		bootstrap_features=False,
		n_jobs=N_JOBS
	)
	print("SVC BAGGING")
	avg_rocauc, confusions = test_model(ScikitModel(
					bagging_svc,
					params=bag_svm_params,
					random_search=True,
					n_iter=25,
					verbose=1),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'svc_bag' not in rocaucs_dict:
		rocaucs_dict['svc_bag'] = []
	rocaucs_dict['svc_bag'].append(avg_rocauc)

	if 'svc_bag' not in confusions_dict:
		confusions_dict['svc_bag'] = []
	confusions_dict['svc_bag'].append(confusions)

	print()


	### Voting Ensemble ###
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
	print("LR-SVC-XGB VOTING ENSEMBLE")
	avg_rocauc, confusions = test_model(ScikitModel(
					eclf,
					eclf_params,
					random_search=True, 
					n_iter=25,
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state)

	if 'voting_clf' not in rocaucs_dict:
		rocaucs_dict['voting_clf'] = []
	rocaucs_dict['voting_clf'].append(avg_rocauc)

	if 'voting_clf' not in confusions_dict:
		confusions_dict['voting_clf'] = []
	confusions_dict['voting_clf'].append(confusions)

	print()

	print()



### SUMMARIZE RESULTS ###
print('\n---------- SUMMARY OF RESULTS ----------\n')
print('Num. random seeds tested: {}'.format(len(RANDOM_STATES)))
print()

print('--- Average of average out-of-sample ROCAUCs: ---')
for model, results_list in rocaucs_dict.items():
	avg_rocauc = np.mean(results_list)
	print('{}: {}'.format(model, avg_rocauc))
print()

print('--- Average confusion info: ---')
for model, results_list in confusions_dict.items():
	print('{} results:'.format(model))
	avg_confusion = np.mean(results_list, axis=0)
	explain_confusion(avg_confusion)
	print()

with open('results_json.txt', 'w') as resultsfile:
	all_results = {
		'roc_results': rocsaucs_dict,
		'confusion_results': confusions_dict
	}
    json.dump(all_results, resultsfile)




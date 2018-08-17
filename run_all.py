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

# from stanford_kd_algorithm import TwoStageModel

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
import xgboost as xgb

# from deepnet.deep_model import DeepKDModel
# from xgbst.xgboost_model import XGBoostKDModel
from model_helpers.models import *

from preprocess import load_data

# Ignore 'Truth value of an array is ambiguous' warning bug: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# Beta for fbeta_score
BETA = 1.5 # 0-1 favors precision, >1 (up to infinity) favors recall
CLASS_WEIGHT = "none" # set to "none" or "balanced"
USE_SMOTE = False
RANDOM_STATES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90007, 2018, 525, 777, 16, 99, 2048, 1880, 42]
N_JOBS = 1
ALLOW_INDETERMINATES = True
CALIBRATION_SET_SIZE = 0.5 # how much of train-set to use for risk-calibration (FC-KD thresholds)
REDUCED_FEATURES = False
VERBOSE = True

# Load expanded dataset
x, y, ids = load_data.load_expanded(one_hot=False, fill_mode='mean', reduced_features=REDUCED_FEATURES, return_pandas=True)

rocaucs_dict = {}
confusions_dict = {}

### RUN EXPERIMENTS ###
for random_state in RANDOM_STATES:

	print("------- RANDOM STATE: {} -------".format(random_state))
	print("")

	# Manually set random seed at the start, in case random_state isn't passed into functions later on
	np.random.seed(random_state)

	#### Stanford Algorithm (WITH SUBCOHORTING) ####
	print("STANFORD KD ALGORITHM")
	print("ABOUT TO GET DESTROYED BY USC")
	print("#BEATtheFARM --- Fight On!")
	stage1 = ScikitModel(LinearDiscriminantAnalysis(), params={}, random_search=False, n_iter=1)
	rf_params = {
	   'n_estimators': [300],
	   'max_features': [1/3]
	}
	stage2 = SubcohortModel(ScikitModel(RandomForestClassifier(), params=rf_params, random_search=False, n_iter=1))
	avg_rocauc, confusions = test_2stage_model(TwoStageModel(
					stage1, stage2,
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'stanford' not in rocaucs_dict:
		rocaucs_dict['stanford'] = []
	rocaucs_dict['stanford'].append(avg_rocauc)

	if 'stanford' not in confusions_dict:
		confusions_dict['stanford'] = []
	confusions_dict['stanford'].append(confusions)

	print()

	#### Stanford Algorithm (WITHOUT SUBCOHORTING) ####
	print("STANFORD KD ALGORITHM (NO SUBCOHORTING)")
	stage1 = ScikitModel(LinearDiscriminantAnalysis(), params={}, random_search=False, n_iter=1)
	rf_params = {
	   'n_estimators': [300],
	   'max_features': [1/3]
	}
	stage2 = ScikitModel(RandomForestClassifier(), params=rf_params, random_search=False, n_iter=1)
	avg_rocauc, confusions = test_2stage_model(TwoStageModel(
					stage1, stage2,
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'stanford_no_subcohort' not in rocaucs_dict:
		rocaucs_dict['stanford_no_subcohort'] = []
	rocaucs_dict['stanford_no_subcohort'].append(avg_rocauc)

	if 'stanford_no_subcohort' not in confusions_dict:
		confusions_dict['stanford_no_subcohort'] = []
	confusions_dict['stanford_no_subcohort'].append(confusions)

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
				calibration_set_size=CALIBRATION_SET_SIZE,
				random_state=random_state,
				return_val='roc_confusion',
				verbose=VERBOSE)

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
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'svc' not in rocaucs_dict:
		rocaucs_dict['svc'] = []
	rocaucs_dict['svc'].append(avg_rocauc)

	if 'svc' not in confusions_dict:
		confusions_dict['svc'] = []
	confusions_dict['svc'].append(confusions)

	print()


	### Random Forest ###
	rf_params = {
	   'n_estimators': [300],
	   'max_features': [1/3]
	}
	print("RANDOM FOREST")
	avg_rocauc, confusions = test_model(ScikitModel(RandomForestClassifier(), 
						params=rf_params,
						random_search=False,
						n_iter=1,
						scoring='roc_auc',
						verbose=True),
					x, y,
					allow_indeterminates=ALLOW_INDETERMINATES,
					random_state=random_state,
					calibration_set_size=CALIBRATION_SET_SIZE,
					return_val='roc_confusion',
					verbose=VERBOSE)

	if 'rf' not in rocaucs_dict:
		rocaucs_dict['rf'] = []
	rocaucs_dict['rf'].append(avg_rocauc)

	if 'rf' not in confusions_dict:
		confusions_dict['rf'] = []
	confusions_dict['rf'].append(confusions)

	print()


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
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

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
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'lr_bag' not in rocaucs_dict:
		rocaucs_dict['lr_bag'] = []
	rocaucs_dict['lr_bag'].append(avg_rocauc)

	if 'lr_bag' not in confusions_dict:
		confusions_dict['lr_bag'] = []
	confusions_dict['lr_bag'].append(confusions)

	print()


	# ### Bagging SVC ###
	# bag_svm_params = {
	# 	'base_estimator__C': np.logspace(-3, 2, 100),
	# 	'base_estimator__gamma': np.logspace(-3, 2, 100),
	# 	'base_estimator__kernel': ['linear', 'rbf', 'poly'],
	# 	'base_estimator__probability': [True, False],
	# 	'n_estimators': randint(5, 50),
	# 	"max_samples": np.logspace(-0.9, 0, 100),
	# 	"max_features": randint(10, x.shape[1])
	# }
	# bagging_svc = BaggingClassifier(
	# 	base_estimator=SVC(),
	# 	bootstrap=True,
	# 	bootstrap_features=False,
	# 	n_jobs=N_JOBS
	# )
	# print("SVC BAGGING")
	# avg_rocauc, confusions = test_model(ScikitModel(
	# 				bagging_svc,
	# 				params=bag_svm_params,
	# 				random_search=True,
	# 				n_iter=25,
	# 				verbose=1),
	# 			x, y,
	# 			allow_indeterminates=ALLOW_INDETERMINATES,
	# 			random_state=random_state,
	# 			calibration_set_size=CALIBRATION_SET_SIZE,
	# 			return_val='roc_confusion')

	# if 'svc_bag' not in rocaucs_dict:
	# 	rocaucs_dict['svc_bag'] = []
	# rocaucs_dict['svc_bag'].append(avg_rocauc)

	# if 'svc_bag' not in confusions_dict:
	# 	confusions_dict['svc_bag'] = []
	# confusions_dict['svc_bag'].append(confusions)

	# print()


	### Voting Ensemble (OPTIMIZE FOR ACCURACY) ###
	# clf1 = SVC(probability=True)
	# clf2 = LogisticRegression()
	# clf3 = xgb.XGBClassifier(n_jobs=N_JOBS)

	# eclf = VotingClassifier(
	#     estimators=[
	#     	('svm', clf1), 
	#     	('lr', clf2),
	#     	('xgb', clf3)
	#     ],
	#     voting='soft',
	#     n_jobs=N_JOBS
	# )
	# eclf_params = {
	#     'svm__C': np.logspace(-3, 2, 100),
	# 	'svm__gamma': np.logspace(-3, 2, 100),
	# 	'svm__kernel': ['rbf', 'poly'],
	#     'lr__C': np.logspace(-3, 2, 100),
	# 	'xgb__n_estimators': randint(50, 500),
	# 	'xgb__max_depth': randint(3, 10),
	# 	'xgb__learning_rate': np.logspace(-2, 0, 100),
	# 	'xgb__min_child_weight': randint(1, 5),
	# 	'xgb__subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
	# 	'xgb__colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	# }
	# print("LR-SVC-XGB VOTING ENSEMBLE (ACCURACY)")
	# avg_rocauc, confusions = test_model(ScikitModel(
	# 				eclf,
	# 				eclf_params,
	# 				random_search=True, 
	# 				scoring='accuracy',
	# 				n_iter=25,
	# 				verbose=True),
	# 			x, y,
	# 			allow_indeterminates=ALLOW_INDETERMINATES,
	# 			random_state=random_state,
	# 			calibration_set_size=CALIBRATION_SET_SIZE,
	# 			return_val='roc_confusion',
	# 			verbose=VERBOSE)

	# if 'voting_clf_accuracy' not in rocaucs_dict:
	# 	rocaucs_dict['voting_clf_accuracy'] = []
	# rocaucs_dict['voting_clf_accuracy'].append(avg_rocauc)

	# if 'voting_clf_accuracy' not in confusions_dict:
	# 	confusions_dict['voting_clf_accuracy'] = []
	# confusions_dict['voting_clf_accuracy'].append(confusions)

	# print()


	### Voting Ensemble (OPTIMIZE FOR ROCAUC) ###
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
	print("LR-SVC-XGB VOTING ENSEMBLE (ROCAUC)")
	avg_rocauc, confusions = test_model(ScikitModel(
					eclf,
					eclf_params,
					random_search=True, 
					scoring='roc_auc',
					n_iter=25,
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'voting_clf_rocauc' not in rocaucs_dict:
		rocaucs_dict['voting_clf_rocauc'] = []
	rocaucs_dict['voting_clf_rocauc'].append(avg_rocauc)

	if 'voting_clf_rocauc' not in confusions_dict:
		confusions_dict['voting_clf_rocauc'] = []
	confusions_dict['voting_clf_rocauc'].append(confusions)

	print()


	### LDA --> VOTING-ENSEMBLE 2-STAGE MODEL (OPTIMIZE FOR ACCURACY) ###
	# Stage 1: LDA
	# stage1 = LinearDiscriminantAnalysis()
	
	# # Stage 2: LR-SVC-XGB Voting Classifier
	# clf1 = SVC(probability=True)
	# clf2 = LogisticRegression()
	# clf3 = xgb.XGBClassifier(n_jobs=N_JOBS)
	# eclf = VotingClassifier(
	#     estimators=[
	#     	('svm', clf1), 
	#     	('lr', clf2),
	#     	('xgb', clf3)
	#     ],
	#     voting='soft',
	#     n_jobs=N_JOBS
	# )
	# eclf_params = {
	#     'svm__C': np.logspace(-3, 2, 100),
	# 	'svm__gamma': np.logspace(-3, 2, 100),
	# 	'svm__kernel': ['rbf', 'poly'],
	#     'lr__C': np.logspace(-3, 2, 100),
	# 	'xgb__n_estimators': randint(50, 500),
	# 	'xgb__max_depth': randint(3, 10),
	# 	'xgb__learning_rate': np.logspace(-2, 0, 100),
	# 	'xgb__min_child_weight': randint(1, 5),
	# 	'xgb__subsample': np.logspace(-0.3, 0, 100), # (~0.5 - 1.0)
	# 	'xgb__colsample_bytree': np.logspace(-0.3, 0, 100) # (~0.5 - 1.0)
	# }
	# stage2 = RandomizedSearchCV(eclf, eclf_params, cv=5, n_iter=25, scoring='accuracy', verbose=1, n_jobs=1)

	# print('LDA + 3-WAY-VOTING-CLASSIFIER 2-STAGE ENSEMBLE (ACCURACY)')

	# avg_rocauc, confusions = test_2stage_model(TwoStageModel(
	# 				stage1, stage2,
	# 				verbose=True),
	# 			x, y,
	# 			allow_indeterminates=ALLOW_INDETERMINATES,
	# 			random_state=random_state,
	# 			calibration_set_size=CALIBRATION_SET_SIZE,
	# 			return_val='roc_confusion',
	# 			verbose=VERBOSE)

	# if 'lda_voting_2stage_accuracy' not in rocaucs_dict:
	# 	rocaucs_dict['lda_voting_2stage_accuracy'] = []
	# rocaucs_dict['lda_voting_2stage_accuracy'].append(avg_rocauc)

	# if 'lda_voting_2stage_accuracy' not in confusions_dict:
	# 	confusions_dict['lda_voting_2stage_accuracy'] = []
	# confusions_dict['lda_voting_2stage_accuracy'].append(confusions)

	# print()


	### LDA --> VOTING-ENSEMBLE 2-STAGE MODEL (0PTIMIZE FOR ROCAUC) ###
	# Stage 1: LDA
	stage1 = ScikitModel(LinearDiscriminantAnalysis(), params={}, random_search=False, n_iter=1)
	
	# Stage 2: LR-SVC-XGB Voting Classifier
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
	stage2 = ScikitModel(eclf, eclf_params, random_search=True, scoring='roc_auc', n_iter=25, verbose=True)

	print('LDA + 3-WAY-VOTING-CLASSIFIER 2-STAGE ENSEMBLE (ROCAUC)')

	avg_rocauc, confusions = test_2stage_model(TwoStageModel(
					stage1, stage2,
					verbose=True),
				x, y,
				allow_indeterminates=ALLOW_INDETERMINATES,
				random_state=random_state,
				calibration_set_size=CALIBRATION_SET_SIZE,
				return_val='roc_confusion',
				verbose=VERBOSE)

	if 'lda_voting_2stage_rocauc' not in rocaucs_dict:
		rocaucs_dict['lda_voting_2stage_rocauc'] = []
	rocaucs_dict['lda_voting_2stage_rocauc'].append(avg_rocauc)

	if 'lda_voting_2stage_rocauc' not in confusions_dict:
		confusions_dict['lda_voting_2stage_rocauc'] = []
	confusions_dict['lda_voting_2stage_rocauc'].append(confusions)

	print()


	# ### LDA --> SVC 2-STAGE MODEL ###
	# stage1 = LinearDiscriminantAnalysis()
	# stage2 = SVC(kernel='rbf', probability=True)
	# print('LDA + SVC 2-STAGE ENSEMBLE')

	# avg_rocauc, confusions = test_2stage_model(TwoStageModel(
	# 				stage1, stage2,
	# 				verbose=True),
	# 			x, y,
	# 			allow_indeterminates=ALLOW_INDETERMINATES,
	# 			random_state=random_state,
	# 			calibration_set_size=CALIBRATION_SET_SIZE,
	# 			return_val='roc_confusion')

	# if 'lda_svc_2stage' not in rocaucs_dict:
	# 	rocaucs_dict['lda_svc_2stage'] = []
	# rocaucs_dict['lda_svc_2stage'].append(avg_rocauc)

	# if 'lda_svc_2stage' not in confusions_dict:
	# 	confusions_dict['lda_svc_2stage'] = []
	# confusions_dict['lda_svc_2stage'].append(confusions)

	# print()



	print()



### SUMMARIZE RESULTS ###
print('\n---------- SUMMARY OF RESULTS ----------\n')
print('Num. random seeds tested: {}'.format(len(RANDOM_STATES)))
print('Reduced feature-set: ', REDUCED_FEATURES)
print('Allow indeterminates: ', ALLOW_INDETERMINATES)
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
	explain_confusion(avg_confusion, indeterminates=ALLOW_INDETERMINATES)
	print()

with open('results_json.txt', 'w') as resultsfile:
	all_results = {
		'roc_results': rocaucs_dict,
		'confusion_results': confusions_dict
	}
	json.dump(all_results, resultsfile)




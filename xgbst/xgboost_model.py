# # KD Diagnosis: XGBoost
# ---
# [XGBoost Documentation](http://xgboost.readthedocs.io/en/latest/python/python_api.html#)
# 1. Load in KD data
# 2. Set XGB hyperparameters
# 3. Train XGB model
# 4. Show feature importance
# 5. Plot XGB tree
# 6. **TODO:** hyperparameter optimization (see: skopt)

from preprocess import load_data
from sklearn import preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import fbeta_score
from skopt import gp_minimize
import numpy as np
from sklearn.model_selection import StratifiedKFold

class XGBoostKDModel:
	def __init__(self, max_depth=5, eta=0.3, objective='binary:logistic', eval_metric='auc',
			num_round=20, verbose=False):
		# Set Hyperparameters
    	# 	Docs: https://xgboost.readthedocs.io/en/latest/parameter.html
		self.param = {
		    'max_depth': max_depth, 
		    'eta': eta, 
		    'objective': objective, 
		    'eval_metric': eval_metric,
		    'silent': 1 - verbose
		}
		self.num_round = num_round
		self.verbose = verbose

		# Get Feature Names
		f = open('data/kd_dataset.pkl','rb')
		x_tr, _, _, _ = pkl.load(f)
		self.feature_names = list(x_tr)


	def train_test(self, x_train, x_test, y_train, y_test):

		# Create Data Matrices
		dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=self.feature_names)
		dtest = xgb.DMatrix(x_test, label=y_test, feature_names=self.feature_names)

		evallist = [(dtest, 'eval'), (dtrain, 'train')]
		# Train model
		self.bst = xgb.train(self.param, dtrain, num_boost_round=self.num_round, evals=evallist, verbose_eval=self.verbose)

		# Draw stuff
		# if self.verbose:
		# # Plot feature importances
		# 	xgb.plot_importance(self.bst)
		# 	plt.show()

		# 	# Draw tree diagram
		# 	TREE_NUM = 0
		# 	xgb.plot_tree(self.bst, num_trees=TREE_NUM)
		# 	fig = plt.gcf()
		# 	fig.set_size_inches(150, 100)
		# 	plt.savefig('xgb-sample-tree.pdf')

		return (np.array(self.bst.predict(dtest)) > 0.5).astype(int)


	# # Weighted score of precision/recall
		
	# 	# Takes in 1 specific x_test/y_test fold
	# def fbeta(self, x_test, y_test, beta=1):# Calculate y_pred
	# 	# Create Data Matrices
	# 	dtest = xgb.DMatrix(x_test, label=y_test, feature_names=self.feature_names)

	# 	y_pred_binary = (np.array(self.bst.predict(dtest)) > 0.5).astype(int)
	# 	# print('y_test: ', y_test)
	# 	return fbeta_score(y_test, y_pred_binary, beta)

	# Train model and return fbeta score averaged over k folds
		# See: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
	def kfold_fbeta(self, x, y, k=5, beta=1):
		fbetas = []
		kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=90007)
		for train_idx, test_idx in kf.split(x, y):
			# Get 1 fold
			x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]

			# Train and test on fold
			y_pred = self.train_test(x_train, x_test, y_train, y_test)
			fbeta = fbeta_score(y_test, y_pred, beta)
			fbetas.append(fbeta)

		# Average over fbetas
		return np.mean(fbetas)


	# Optimize model hyperparameters based on fbeta_score, save optimal parameters in member vars
		# This represents one OUTER CV-training loop
	def optimize_hyperparameters(self, x_train, y_train, beta=1, num_calls=100, random_state=None, k=5):

		# Hyperparameters
			# Eta (learning rate): 0.01-0.2
			# max_depth (3-10)
			# min_child_weight (1-10)
			# subsample (0.5-1)
			# colsample_bytree: (0.5-1)
			# early_stopping_rounds: 20

		# Optimization objective for skopt: returns negated fbeta score
		# Input: tuple of hyperparameters
		def skopt_objective(params):
			# Unpack hyperparameters
			eta, max_depth, min_child_weight, subsample, colsample_bytree, num_round = params

			# Set hyperparameter
			self.param['eta'] = eta
			self.param['max_depth'] = max_depth
			self.param['min_child_weight'] = min_child_weight
			self.param['subsample'] = subsample
			self.param['colsample_bytree'] = colsample_bytree

			# Return negated fbeta_score (minimize negative fbeta --> maximize fbeta)
			return -self.kfold_fbeta(x_train, y_train, k=k, beta=beta)

		# Define hyperparameter space
		hyperparam_space = [
			(0.01, 0.3, 'log-uniform'), # eta
			(3, 10), # max_depth
			(1, 10), # min_child_weight
			(0.5, 1.0), # subsample
			(0.5, 1.0), # colsample_bytree
			(5, 50) # num boosting rounds
		]

		# Call skopt to run smart "grid search"
		opt_results = gp_minimize(skopt_objective, hyperparam_space,
						n_calls=num_calls,
						random_state=random_state,
						verbose=self.verbose)
		# Unpack results
		optimal_hyperparams = opt_results.x
		optimal_score = -opt_results.fun

		opt_eta = optimal_hyperparams[0]
		opt_max_depth = optimal_hyperparams[1]
		opt_min_child_weight = optimal_hyperparams[2]
		opt_subsample = optimal_hyperparams[3]
		opt_colsample_bytree = optimal_hyperparams[4]
		opt_num_round = optimal_hyperparams[5]

		# Print hyperparameter optimization results
		if self.verbose:
			print()
			print('----- HYPERPARAMETER OPTIMIZATION RESULTS -----')
			print('Optimal average fbeta score: ', optimal_score)
			print('Optimal eta: ', opt_eta)
			print('Optimal max_depth: ', opt_max_depth)
			print('Optimal min_child_weight: ', opt_min_child_weight)
			print('Optimal subsample: ', opt_subsample)
			print('Optimal colsample_bytree: ', opt_colsample_bytree)
			print('Optimal num_boost_round: ', opt_num_round)

		# Update model hyperparameter to optimal	
		self.param['eta'] = opt_eta
		self.param['max_depth'] = opt_max_depth
		self.param['min_child_weight'] = opt_min_child_weight
		self.param['subsample'] = opt_subsample
		self.param['colsample_bytree'] = opt_colsample_bytree

		# Train 1 last time using optimal hyperparams
		print()
		print('Re-training with optimal hyperparameters...')
		dtrain_complete = xgb.DMatrix(x_train, label=y_train, feature_names=self.feature_names)
		evallist = [(dtrain_complete, 'all_train')]
		self.bst = xgb.train(self.param, dtrain_complete, num_boost_round=opt_num_round, evals=evallist, verbose_eval=self.verbose)




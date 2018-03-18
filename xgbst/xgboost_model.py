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

	def train_test(self, x_train, x_test, y_train, y_test):
		# Get Feature Names
		f = open('data/kd_dataset.pkl','rb')
		x_tr, _, _, _ = pkl.load(f)
		feature_names = list(x_tr)
		# Create Data Matrices
		dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
		dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)

		evallist = [(dtest, 'eval'), (dtrain, 'train')]
		# Train model
		bst = xgb.train(self.param, dtrain, num_boost_round=self.num_round, evals=evallist, verbose_eval=self.verbose)

		if self.verbose:
		# Plot feature importances
			xgb.plot_importance(bst)
			plt.show()

			# Draw tree diagram
			TREE_NUM = 0
			xgb.plot_tree(bst, num_trees=TREE_NUM)
			fig = plt.gcf()
			fig.set_size_inches(150, 100)
			plt.savefig('xgb-sample-tree.pdf')

		import numpy as np
		return (np.array(bst.predict(dtest)) > 0.5).astype(int)




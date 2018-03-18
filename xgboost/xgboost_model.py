# # KD Diagnosis: XGBoost
# ---
# [XGBoost Documentation](http://xgboost.readthedocs.io/en/latest/python/python_api.html#)
# 1. Load in KD data
# 2. Set XGB hyperparameters
# 3. Train XGB model
# 4. Show feature importance
# 5. Plot XGB tree
# 6. **TODO:** hyperparameter optimization (see: skopt)

# Imports
import sys
sys.path.append('../') # Make parent folder visible
from preprocess import load_data
from sklearn import preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle as pkl

# Load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=False)

# Preprocessing - XGBoost should be invariant to this
# scaler = preprocessing.StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# Get feature names
f = open('../data/kd_dataset.pkl','rb')
x_train, x_test, y_train, y_test = pkl.load(f)
feature_names = list(x_train)

# Create Data Matrices
dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)

# Set Hyperparameters
    # Docs: https://xgboost.readthedocs.io/en/latest/parameter.html
param = {
    'max_depth': 5, 
    'eta': 0.3, 
    'objective': 'binary:logistic', 
    'eval_metric': 'auc'
}

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 20

# Train model
bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=evallist)

# Plot feature importances
xgb.plot_importance(bst)
plt.show()

# Draw tree diagram
TREE_NUM = 0
xgb.plot_tree(bst, num_trees=TREE_NUM)
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.savefig('xgb-sample-tree.pdf')


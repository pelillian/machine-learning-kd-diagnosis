# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# run_all.py

# This script runs each model with different parameters

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

from deepnet.deep_model import DeepKDModel
from preprocess import load_data

# load data
x_train, x_test, y_train, y_test = load_data.load(one_hot=True, fill_mode='mean')

deep = DeepKDModel()
deep.train(x_train, y_train)
print(deep.test(x_test, y_test))
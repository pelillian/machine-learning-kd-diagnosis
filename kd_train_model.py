import pickle as pkl
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------
# MACHINE LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# kd_train_model.py

# This script trains the model on the KD dataset

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

f = open('data/kd_dataset.pkl','rb')
x_train, x_test, y_train, y_test = pkl.load(f)
f.close()

print(x_train, x_test, y_train, y_test)

import os
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------------------
# MACHINE LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# kd_create_database.py

# This script creates a database from the excel file

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

# params
test_set_size = 0.3

# studies file
data = pd.read_excel(open('data/KD-FC-Peter-alg-BLINDED-set1-20171229.xlsx', 'rb'), sheetname='Peter set 1- training set')

del data['peternum']

x = data.copy()
del x['label']

y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size)

dataset = [x_train, x_test, y_train, y_test]

f = open('data/kd_dataset.pkl','wb')
pkl.dump(dataset, f)
f.close()


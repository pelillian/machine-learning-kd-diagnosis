# -------------------------------------------------------------------------------------------------
# DEEP LEARNING FOR KAWASAKI DISEASE DIAGNOSIS

# kd_create_database.py

# This script creates a database from the patient data file

# by Peter Lillian
# -------------------------------------------------------------------------------------------------

import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# parameters
test_set_size = 0.3
random_state = 222 # set this to None for random train/test split

# patient data file
data = pd.read_excel(open('data/KD-FC-Peter-alg-BLINDED-set1-20171229.xlsx', 'rb'), sheetname='Peter set 1- training set')

# replace missing data with nan so that we can handle it easier later
data.replace('NA', np.nan)

del data['peternum']
del data['signESR']
del data['signCRP']
del data['signALT']
del data['signGGT']

x = data.copy()
del x['label']

y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size, random_state=random_state)

dataset = [x_train, x_test, y_train, y_test]

f = open('data/kd_dataset.pkl','wb')
pkl.dump(dataset, f)
f.close()


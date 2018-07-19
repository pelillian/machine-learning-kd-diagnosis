import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read in Excel sheet
df_test = pd.read_excel(open('../data/testing/KD-FC-Peter-alg-BLINDED-multisite-test-set-20180614.xlsx', 'rb'), 
                        sheetname='Sheet1',
                        na_values='NA')


# Drop some extraneous features
drop_features = ['phgb', 'fever']
df_cleaned = df_test.drop(drop_features, axis=1)

# Drop features with lots of NaNs (or 'sign' features)
drop_features_2 = ['pesrsign', 'pcrpsign', 'paltsign', 'pggtsign']
df_cleaned = df_cleaned.drop(drop_features_2, axis=1)

# Pull out peternum (ID) column)
df_id = df_cleaned['pnum2 (Peter test set, 6/14/18)']
df_feat = df_cleaned.drop(['pnum2 (Peter test set, 6/14/18)'], axis=1)

print("Features: ", list(df_feat.columns))
print("Num. features: ", len(df_feat.columns))

# See which features have NaNs -- and if so, how many
print("Number of NaNs:")
print(df_feat.isnull().sum())

# Dataset tuple
dataset = (df_feat, df_id)

f = open('../data/kd_dataset_test.pkl','wb')
pkl.dump(dataset, f)
f.close()

print("Successfully created test dataset file!")
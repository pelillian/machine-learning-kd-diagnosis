import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read in Excel sheets
kd_df = pd.read_excel(open('../data/KD-FC-mrg-peter-alg-expanded-trainingset-20180315.xlsx', 'rb'), 
                        sheetname='KD expanded training set 2018')
fc_df = pd.read_excel(open('../data/KD-FC-mrg-peter-alg-expanded-trainingset-20180315.xlsx', 'rb'), 
                        sheetname='FC expanded training set 2018')
df = pd.concat([kd_df, fc_df])

# Get inner join (features that only appear in both tables)
df_inner = pd.concat([kd_df, fc_df], join='inner')

# Only include numeric features
df_numeric = df_inner.select_dtypes(include=[np.number])
df_numeric.columns

# Drop some extraneous features
drop_features = ['peternum (12/29/17)', 'peternum cohort 12/29/17: 1=training, 2=validation',
                'pnum cohort 3/15/18: 1=training, 2=validation', 'age_lab', 'phgb', 'phct', 'fever']
df_cleaned = df_numeric.drop(drop_features, axis=1)

# Split training features/label
df_train = df_cleaned.drop(['label'], axis=1)
df_label = df_cleaned['label']

# Drop features with lots of NaNs (or 'sign' features)
drop_features_2 = ['caa', 'pesrsign', 'pcrpsign', 'paltsign', 'pggtsign', 'psodium', 'bnp']
df_train = df_train.drop(drop_features_2, axis=1)

# Pull out peternum (ID) column)
df_id = df_train['pnum (3/15/18)']
df_train = df_train.drop(['pnum (3/15/18)'], axis=1)

print("Training features: ", list(df_train.columns))
print("Num. training features: ", len(df_train.columns))

# See which features have NaNs -- and if so, how many
print("Number of NaNs:")
print(df_train.isnull().sum())

# Dataset tuple
dataset = (df_train, df_label, df_id)

f = open('../data/kd_dataset_expanded.pkl','wb')
pkl.dump(dataset, f)
f.close()

print("Successfully created expanded dataset file!")
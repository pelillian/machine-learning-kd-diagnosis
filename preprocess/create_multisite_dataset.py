import pickle as pkl
import pandas as pd
import numpy as np

# Read spreadsheet
df = pd.read_excel(open('../data/KD-FC-multisite-USC-alg-20181121-training-set.xlsx', 'rb'), 
                        sheet_name='For Peter-67% training set')

# Get IDs, Labels
ids = df['pnum2']
labels = df['True Record type: 0= FC, 1=KD']

# 18 features
reduced_features = ['illday', 'rash', 'redeyes', 'redplt', 'clnode', \
		'redhands', 'pwbc', 'ppolys', 'pbands', 'plymphs', 'pmonos',\
		'peos', 'pesr', 'pcrp', 'pplts', 'palt', 'pggt', 'zhemo']

# Only keep reduced features
df_cleaned = df[reduced_features]

# # Features to drop - TODO
# drop_features = ['pnum2', 'cohort (1=train,2=test)', 'True Record type: 0= FC, 1=KD',
#                 'pnum cohort 3/15/18: 1=training, 2=validation', 'age_lab', 'phgb', 'phct', 'fever',
#                 'caa', 'pesrsign', 'pcrpsign', 'paltsign', 'pggtsign', 'psodium', 'bnp']


# import pdb; pdb.set_trace();

dataset = (df_cleaned, labels, ids)

f = open('../data/kd_dataset_multisite.pkl','wb')
pkl.dump(dataset, f)
f.close()
print('Successfully created multisite dataset file! ({} rows)'.format(len(df_cleaned)))
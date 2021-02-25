#%% imports
from knn_sklearn import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from itertools import combinations

#%% load data
df = pd.read_csv('data/cleveland.csv')
df_val = pd.read_csv('data/cleveland-test-sample.csv')

# drop rows where thal and ca are '?'
df = df[df['thal'] != '?']
df = df[df['ca'] != '?']

# convert thal and ca to float
df['thal'] = df['thal'].astype('float64')
df['ca'] = df['ca'].astype('float64')
df['disease'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

df_val['ca'] = df['ca'].astype('float64')

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
# print(df_val.dtypes)

#%% k nearest neighbors all column combinations

import time

# all combinations of columns
combos = []
for i in range(1, len(columns)+1):
    combos.extend(list(x) for x in combinations(columns, i) )
print(len(combos), 'combinations')

results = []
best_f1 = 0
best_name = ''
for i, cols in enumerate(combos):
    name = '_'.join(cols)
    X = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
    y = df['disease'].to_numpy() 
    
    start = time.time()
    for k in range(1, 23, 2):
        precision, recall, f1, support = \
            k_fold_validation(10, k, X, y)
        results.append((name, k, precision, recall, f1, support))
        if (f1 > best_f1):
            best_f1 = f1
            best_name = name
    s = time.time() - start
    print('%50s'%name, "\t%.5ss"%s, '%f'%((len(combos)-i)*s/3600), '%f'%(i/(len(combos))), best_name, best_f1)


#%% turn results into data frame
resultdf = pd.DataFrame(results, columns=['columns', 'k', 'precision', 'recall', 'f1', 'support'])
resultdf['dims'] = resultdf['columns'].str.split('_').apply(lambda x: len(x))
resultdf.to_csv('data/AllCombinations.csv', index=False)

#%% load all combinations results
resultdf = pd.read_csv('data/AllCombinations.csv')

#%% get top performing parameters
col = 'f1'
best_accuracy = resultdf.sort_values(col).iloc[-1][col]
display(resultdf.sort_values(col, ascending=False).head(30))
display(resultdf[resultdf[col] == best_accuracy])

#%% precision vs recall
plt.figure()
sns.scatterplot(x='precision', y='recall', data=resultdf[resultdf.f1 >= .79])
plt.title('Precision vs Recall of classifiers with F1 >= .8')
plt.show()

#%% best performer - k fold validation
cols = ['cp', 'exang', 'ca', 'thal']
cols = ['ca', 'cp', 'thal']
k = 11 # choose odd k so there is never a tie

X = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
y = df['disease'].to_numpy() 

precision, recall, f1, support = test_columns(df, cols, 'disease', k, k_fold_validation)

print('precision  recall    f1        support')
print('%5f'%precision, '  %5f'%recall, ' %5f'%f1, ' %5f'%support)

#%% Best performer - validation set
cols = ['cp', 'exang', 'ca', 'thal']
cols = ['ca', 'cp', 'thal']
k = 11 # choose odd k so there is never a tie

X_train  = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
y_train = df['disease'].to_numpy() 

X_validate = np.array([row.to_numpy() for index, row in df_val[cols].iterrows()])
y_validate = df_val['disease'].to_numpy()

classify = build_classifier(k, X_train, y_train)
precision, recall, f1, support = validate(classify, X_validate, y_validate)

print('precision  recall    f1        support')
print('%5f'%precision, '  %5f'%recall, ' %5f'%f1, ' %5f'%support)
 
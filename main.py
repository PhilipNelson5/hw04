#%% imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

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

#%% functions
def validate(classify, X_test, y_test):
    """validate a classifier

    Args:
        classify (function(list(float))): classifier that accepts
            a list of vectors and returns a list of labels
        X_test (list(list(float))): list of known vectors
        y_test (list(float)): list of known labels
    """
    y_predictions = classify(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, y_predictions, labels=[1])
    return p[0], r[0], f1[0], s[0]

def build_classifier(k, X_train, y_train):
    """build a k nearest neighbors classifier

    Args:
        k ([int]): number of neighbors
        X_train (list(list(float))): list of known vectors
        y_train (list(float)): list of known labels

    Returns:
        function(list(list(float))) -> list(float): a knn classifier which
            classifies a list of vecors
    """
    fit = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto') \
        .fit(X_train)

    def classify(X_test):
        _, indices = fit.kneighbors(X_test)
        return [ np.bincount(y_train[ids]).argmax() for ids in indices ]

    return classify


def k_fold_validaiton(k_fold, k, X, y):
    precision_t = 0
    recall_t = 0
    f1_t = 0
    support_t = 0

    for train_ind, test_ind in KFold(k_fold, True).split(X, y):
        X_train = X[train_ind]
        X_test = X[test_ind]
        y_train = y[train_ind]
        y_test = y[test_ind]

        classify = build_classifier(k, X_train, y_train)
        precision, recall, f1, support = validate(classify, X_test, y_test)

        precision_t += precision
        recall_t += recall
        f1_t += f1
        support_t += support

    return precision_t/k_fold, recall_t/k_fold, f1_t/k_fold, support_t/k_fold

def monte_carlo_validation(n, k, X, y):
    precision_t = 0
    recall_t = 0
    f1_t = 0
    support_t = 0

    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        classify = build_classifier(k, X_train, y_train)
        precision, recall, f1, support = validate(classify, X_test, y_test)

        precision_t += precision
        recall_t += recall
        f1_t += f1
        support_t += support

    return precision_t/n, recall_t/n, f1_t/n, support_t/n


def test_columns(cols, label_col, k, validator):
    X = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
    y = df['disease'].to_numpy() 
    return validator(10, k, X, y)


#%% k nearest neighbors

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
            k_fold_validaiton(10, k, X, y)
        results.append((name, k, precision, recall, f1, support))
        if (f1 > best_f1):
            best_f1 = f1
            best_name = name
    s = time.time() - start
    print('%50s'%name, "\t%.5ss"%s, '%f'%((len(combos)-i)*s/3600), '%f'%(i/(len(combos))), best_name, best_f1)


#%%
resultdf = pd.DataFrame(results, columns=['columns', 'k', 'precision', 'recall', 'f1', 'support'])
resultdf['dims'] = resultdf['columns'].str.split('_').apply(lambda x: len(x))
resultdf.to_csv('data/AllCombinations.csv', index=False)

#%%
resultdf = pd.read_csv('data/AllCombinations.csv')

#%%
col = 'f1'
best_accuracy = resultdf.sort_values(col).iloc[-1][col]
display(resultdf.sort_values(col, ascending=False).head(30))
display(resultdf[resultdf[col] == best_accuracy])
#%%
cols = ['restecg','oldpeak','ca']
cols = ['cp', 'exang', 'ca', 'thal']
cols = ['ca', 'cp', 'thal']
k = 7 # choose odd k so there is never a tie
X = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
y = df['disease'].to_numpy() 

precision, recall, f1, support = test_columns(cols, 'disease', k, k_fold_validaiton)
print(precision, recall, f1, support)

#%% imports
import knn

import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from itertools import combinations
from sklearn.model_selection import train_test_split

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

#%% k nearest neighbors
cols = ['restecg','oldpeak','ca']
k = 8
data = [row.to_numpy() for index, row in df[cols].iterrows()]
labels = df['disease'].to_numpy() 
data = list(zip(data, labels))

valid = [row.to_numpy() for index, row in df_val[cols].iterrows()]
labels_val = df_val['disease'].to_numpy()
valid = list(zip(valid, labels_val))

train, test = train_test_split(data, train_size=.8)
classifier = partial(knn.classify, k, train)
print(knn.validate(classifier, test))
print(knn.validate(classifier, valid))

# for i in range(1, 55, 5):
#     v = validate(partial(classify_knn, i), train, test)
#     print(i, v)

#%%
import time

# all combinations of columns
combs = []
for i in range(3, 5+1):
    combs.extend(list(x) for x in combinations(columns, i) )
print(len(combs), 'combinations')

start = 0 # = combs.index('sex_chol_thal'.split('_')) + 1

results = []
n = 10
for i, cols in enumerate(combs[start:]):
    name = '_'.join(cols)
    data = [row.to_numpy() for index, row in df[cols].iterrows()]
    labels = df['disease'].to_numpy() 
    data = list(zip(data, labels))
    
    start = time.time()
    for k in range(1, 21):
        acc_t = 0
        precision_t = 0
        recall_t = 0
        f1_t = 0
        i = 0
        x = 0
        while i < n:
            train, test = train_test_split(data, train_size=.8, random_state=i+x)
            try:
                accuracy, precision, recall, f1 = \
                    knn.validate(partial(knn.classify, k, train), test)
            except:
                x += 1
                continue
            acc_t += accuracy
            precision_t += precision
            recall_t += recall
            f1_t += f1
            i += 1
        results.append((name, k, acc_t/n, precision_t/n, recall_t/n, f1_t/n))
    s = time.time() - start
    print(name, "\t%.5ss" % s, '%f' % ((len(combs)-i)*s/3600), '%f' % (i/(len(combs))))

#%%
resultdf = pd.DataFrame(results, columns=['columns', 'k', 'accuracy', 'precision', 'recall', 'f1'])
resultdf['dims'] = resultdf['columns'].str.split('_').apply(lambda x: len(x))
resultdf.to_csv('data/AllCombinations.csv', index=False)

#%%
resultdf = pd.read_csv('data/AllCombinations.csv')

#%%
col = 'f1'
best_accuracy = resultdf.sort_values(col).iloc[-1][col]
display(resultdf.sort_values(col, ascending=False).head(15))
display(resultdf[resultdf[col] == best_accuracy])

#%%
plt.figure()
sns.scatterplot(x='precision', y='recall', data=resultdf)
plt.show()

plt.figure()
sns.scatterplot(x='k', y='f1', data=resultdf)
plt.show()
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

# drop rows where thal and ca are '?'
df = df[df['thal'] != '?']
df = df[df['ca'] != '?']

# convert thal and ca to float
df['thal'] = df['thal'].astype('float64')
df['ca'] = df['ca'].astype('float64')

columns = list(df.columns[:-1])

#%% k nearest neighbors
data = [row.to_numpy() for index, row in df[columns].iterrows()]
labels = df['num'].to_numpy() 
data = list(zip(data, labels))
train, test = train_test_split(data, train_size=.8, random_state=0)

# for i in range(1, 55, 5):
#     v = validate(partial(classify_knn, i), train, test)
#     print(i, v)

#%%
import time

# all combinations of columns
combs = []
for i in range(1, len(columns)+1):
    combs.extend(list(x) for x in combinations(columns, i) )
print(len(combs), 'combinations')

start_all = time.time()
results = []
for i, cols in enumerate(combs):
    name = '_'.join(cols)
    data = [row.to_numpy() for index, row in df[cols].iterrows()]
    labels = df['num'].to_numpy() 
    data = list(zip(data, labels))
    train, test = train_test_split(data, train_size=.8, random_state=0)
    
    start = time.time()
    acc_max = 0
    f1_max = 0
    for k in range(1, 21):
        acc, f1 = knn.validate(partial(knn.classify, k, train), test)
        acc_max = max(acc_max, acc)
        f1_max = max(f1_max, f1)
        results.append((name, k, acc, f1))
    s = time.time() - start
    print("%f" % f1_max, "%f" % acc_max, name, "\t%.5ss" % s, '%f' % ((len(combs)-i)*s/3600), '%f' % (i/(len(combs))))
print("total time %f" % s/3600)

#%%
resultdf = pd.DataFrame(results, columns=['columns', 'k', 'accuracy', 'f1'])
resultdf['dims'] = resultdf['columns'].str.split('_').apply(lambda x: len(x))
resultdf.to_csv('data/AllCombinations.csv', index=False)

#%%
resultdf = pd.read_csv('data/AllCombinations.csv')

#%%
best_accuracy = resultdf.sort_values('accuracy').iloc[-1]['accuracy']
display(resultdf[resultdf['accuracy'] == best_accuracy])

#%%
plt.figure()
sns.scatterplot(x='dims', y='accuracy', data=resultdf)
plt.show()

plt.figure()
sns.scatterplot(x='k', y='accuracy', data=resultdf)
plt.show()

#%% imports
from knn_sklearn import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from itertools import combinations
from sklearn.datasets import fetch_openml

#%% load data
mnist = fetch_openml('mnist_784', as_frame=False, cache=True)
X = mnist.data
y = np.vectorize(int)(mnist.target)
# X = np.array([row.to_numpy() for _, row in mnist.data.iterrows()])
# y = mnist.target.to_numpy()

#%%
print(len(X))
print(len(X[0]))

#%% concurrent
import concurrent.futures
from functools import partial

def run(X, Y, k):
  return k_fold_validation(10, k, X, y, lables=list(range(0,10)))

print('k  precision  recall    f1        support')
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
  for precision, recall, f1, support in zip(list(range(1,100,2)), executor.map(partial(run, X, y), list(range(1,100,2)))):
    results.append((k, precision, recall, f1, support))
    print(k, '%5f'%precision, '  %5f'%recall, ' %5f'%f1, ' %5f'%support)

resultdf = pd.DataFrame(results, columns=['k', 'precision', 'recall', 'f1', 'support'])
resultdf.to_csv('data/MnistResults.csv', index=False)

#%%
k = 11 # choose odd k so there is never a tie

print('k  precision  recall    f1        support')
results = []
for k in range(1, 100, 2):
  precision, recall, f1, support = k_fold_validation(10, k, X, y, labels=list(range(0,10)))
  results.append((k, precision, recall, f1, support))
  print(k, '%5f'%precision, '  %5f'%recall, ' %5f'%f1, ' %5f'%support)

resultdf = pd.DataFrame(results, columns=['k', 'precision', 'recall', 'f1', 'support'])
resultdf.to_csv('data/MnistResults.csv', index=False)

#%%
df = pd.read_csv('data/MnistResults.csv')

#%%
sns.scatterplot(x='k', y='precision', data=df, label='Precision')
sns.scatterplot(x='k', y='recall', data=df, label='Recall')
sns.scatterplot(x='k', y='f1', data=df, label = 'F1')
plt.title('Precision, Recall and F1 vs K on MNIST')
plt.xlabel('value')
plt.show()
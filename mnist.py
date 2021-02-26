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

labels = list(range(0,10))

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classify = build_classifier(5, X_train, y_train)
y_predictions = classify(X_test)
p, r, f1, s = precision_recall_fscore_support(y_test, y_predictions, labels=labels)
display(p, r, f1, s)

#%%
precision, recall, f1, support = k_fold_validation(10, k, X, y, labels=labels)
display(k, precision, recall, f1, support)

#%%
k = 11 # choose odd k so there is never a tie

print('k  precision  recall    f1        support')
results = []
for k in range(1, 100, 2):
  precision, recall, f1, support = k_fold_validation(10, k, X, y, labels=labels)
  results.append((k, precision, recall, f1, support))
  print(k, '%5f'%precision, '  %5f'%recall, ' %5f'%f1, ' %5f'%support)

df = pd.DataFrame(results, columns=['k', 'precision', 'recall', 'f1', 'support'])
df.to_csv('data/MnistResults.csv', index=False)

#%%
df = pd.read_csv('data/MnistResults.csv')

#%%
sns.scatterplot(x='k', y='precision', data=df, label='Precision')
sns.scatterplot(x='k', y='recall', data=df, label='Recall')
sns.scatterplot(x='k', y='f1', data=df, label = 'F1')
plt.title('Precision, Recall and F1 vs K on MNIST')
plt.xlabel('k value')
plt.show()
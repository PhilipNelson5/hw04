#%% imports
from knn_sklearn import *
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from itertools import combinations
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

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
k=5
precision, recall, f1, support = k_fold_validation(10, k, X, y, labels=labels)
display(k, precision, recall, f1, support)

#%%
k=5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
classify = build_classifier(k, X_train, y_train)
y_pred = classify(X_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)
display(cm)
plot_confusion_matrix_from_data(y_test, y_pred, labels, path='images/confusion_matrix.pdf')

#%%
print('k  precision  recall    f1        support')
results = []
for k in range(1, 100, 2):
  precision, recall, f1, support = k_fold_validation(10, k, X, y, labels=labels)
  results.append((k, precision, recall, f1, support))
  print(k, '%5f'%precision[0], '  %5f'%recall[0], ' %5f'%f1[0], ' %5f'%support[0])

#%%
# df = pd.DataFrame(results, columns=['k', 'precision', 'recall', 'f1', 'support'])
# df.to_csv('data/MnistResults2.csv', index=False)

#%%
df = pd.read_csv('data/MnistResults2.csv')
display(f1)

#%%
new=[]
for k, f1 in df[['k', 'f1']].iterrows():
  new.append((f1[0], *f1[1]))
f1 = pd.DataFrame(new, columns=['k', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


#%%
for i in range(10):
  sns.lineplot(x='k', y=str(i), data=f1, label=str(i))
plt.title('F1 Score vs k Value')
plt.ylabel('f1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('images/f1_vs_k_mnist.pdf')
plt.show()

#%%
df = pd.read_csv('data/MnistResults.csv')

#%%
sns.scatterplot(x='k', y='precision', data=df, label='Precision')
sns.scatterplot(x='k', y='recall', data=df, label='Recall')
sns.scatterplot(x='k', y='f1', data=df, label = 'F1')
plt.title('Precision, Recall and F1 vs K on MNIST')
plt.xlabel('k value')
plt.show()
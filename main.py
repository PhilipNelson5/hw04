#%% imports
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from functools import partial
from collections import Counter
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
distance = lambda a, b: np.linalg.norm(a-b)

def majority_vote(labels):
    """assumes labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
        for count in vote_counts.values()
        if count == winner_count]
    )
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # try again without the farthest neighbor

def classify_knn(k, points, unknown):
    """each labeled point should be a pair (point, label)"""
    dist_to_unknown = sorted(
       points,
       key=lambda point_label: distance(point_label[0], unknown)
    ) 
    
    k_nearest_labels = [label for _, label in dist_to_unknown[:k]]
    
    return majority_vote(k_nearest_labels)

def validate(classifier, train, test):
    """takes training data and test data and validates the classifier"""
    correct = 0
    for unknown, label in test:
        prediction = classifier(train, unknown)
        if prediction == label: correct += 1
    return correct / len(test)

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
len(combs)

results = []
for i, cols in enumerate(combs):
    name = '_'.join(cols)
    data = [row.to_numpy() for index, row in df[cols].iterrows()]
    labels = df['num'].to_numpy() 
    data = list(zip(data, labels))
    train, test = train_test_split(data, train_size=.8, random_state=0)
    
    start = time.time()
    v_max = 0
    for k in range(1, 21):
        v = validate(partial(classify_knn, k), train, test)
        v_max = max(v_max, v)
        results.append((name, k, v))
        # print(name, k, v)
    s = time.time() - start
    print("%f" % v_max, name, "\t%.5ss" % s, '%f' % ((len(combs)-i)*s/3600), '%f' % (i/(len(combs))))

#%%
sorted(results,
    key=lambda a: a[2],
    reverse=True)[:50]

#%%
# resultdf = pd.DataFrame(results, columns=['columns', 'k', 'accuracy'])
# resultdf['dims'] = resultdf['columns'].str.split('_').apply(lambda x: len(x))

#%%
resultdf = pd.read_csv('data/AllCombinations.csv')
plt.figure()
sns.scatterplot(x='dims', y='accuracy', data=resultdf)
plt.show()

plt.figure()
sns.scatterplot(x='k', y='accuracy', data=resultdf)
plt.show()

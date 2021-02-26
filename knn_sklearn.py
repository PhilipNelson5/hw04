import numpy as np

from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

def validate(classify, X_test, y_test, labels=[1]):
    """validate a classifier

    Args:
        classify (function(list(float))): classifier that accepts
            a list of vectors and returns a list of labels
        X_test (list(list(float))): list of known vectors
        y_test (list(float)): list of known labels
    """
    y_predictions = classify(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, y_predictions, labels=labels)
    return p, r, f1, s

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


def k_fold_validation(k_fold, k, X, y, labels=[1]):
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
        precision, recall, f1, support = validate(classify, X_test, y_test, labels)

        precision_t += precision
        recall_t += recall
        f1_t += f1
        support_t += support

    return precision_t/k_fold, recall_t/k_fold, f1_t/k_fold, support_t/k_fold

def monte_carlo_validation(n, k, X, y, labels=[1]):
    precision_t = 0
    recall_t = 0
    f1_t = 0
    support_t = 0

    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        classify = build_classifier(k, X_train, y_train)
        precision, recall, f1, support = validate(classify, X_test, y_test, labels)

        precision_t += precision
        recall_t += recall
        f1_t += f1
        support_t += support

    return precision_t/n, recall_t/n, f1_t/n, support_t/n


def test_columns(df, cols, label_col, k, validator, labels=[1]):
    X = np.array([row.to_numpy() for index, row in df[cols].iterrows()])
    y = df['disease'].to_numpy() 
    return validator(10, k, X, y, labels=labels)


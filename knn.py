import numpy as np
from collections import Counter

"""function used to calculate the euclidean distance of two vectors"""
distance = lambda a, b: np.linalg.norm(a-b)

def majority_vote(labels):
    """choose the majority label from list of labels sorted by distance

    Args:
        labels ([float]): labels, ordered from nearest to farthest

    Returns:
        [float]: majority label
    """

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

def classify(k, points, unknown):
    """k nearest neighbors classifier

    Args:
        k (int): k
        points ([([float], float)]): list of known (vector, label) tuples
        unknown ([float]): unknown vector to classify

    Returns:
        [float]: classification
    """

    dist_to_unknown = sorted(
       points,
       key=lambda point_label: distance(point_label[0], unknown)
    ) 
    
    k_nearest_labels = [label for _, label in dist_to_unknown[:k]]
    
    return majority_vote(k_nearest_labels)

def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn):
    return tp / (tp + fp)

def recall(tp, fp, fn):
    return tp / (tp + fn)

def f1(tp, fp, fn):
    # p = precision(tp, fn, tp)
    # r = recall(tp, tp, fn)
    # return 2 * p * r / (p + r)
    return tp / (tp + (fp + fn) / 2)

def validate(classifier, test):
    """validate a classifier

    Args:
        classifier (function([float])): classification function that accepts
            a vector and returns a predicted label
        test ([([float], float)]): list of known (vector, label) tuples

    Returns:
        (float, float): classifier accuracy and f1 score
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for unknown, label in test:
        prediction = classifier(unknown)
        if prediction == label: # true
            if prediction == 0: # negative
                tn += 1
            else: # positive
                tp += 1
        else: # false
            if prediction == 0: # negative
                fn += 1
            else: # positive
                fp += 1
                
    return accuracy(tp, fp, fn, tn), f1(tp, fp, fn)

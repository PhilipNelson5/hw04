import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

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
        points ([([float], int)]): list of known (vector, label) tuples
        unknown ([float]): unknown vector to classify

    Returns:
        [int]: classification
    """

    dist_to_unknown = sorted(
       points,
       key=lambda point_label: distance(point_label[0], unknown)
    ) 
    
    k_nearest_labels = [label for _, label in dist_to_unknown[:k]]
    
    return majority_vote(k_nearest_labels)

def validate(classifier, test):
    """validate a classifier

    Args:
        classifier (function([float])): classification function that accepts
            a vector and returns a predicted label
        test ([([float], float)]): list of known (vector, label) tuples

    Returns:
        (float, float, float, float): classifier precision, recall, f1 score, and support
    """
    predictions = []
    labels = []
    for unknown, label in test:
        labels.append(label)
        predictions.append(classifier(unknown))
                
    p, r, f1, s = precision_recall_fscore_support(labels, predictions, labels=[1])
    return p[0], r[0], f1[0], s[0]

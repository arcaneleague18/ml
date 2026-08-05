from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from math import log2

def entropy(c):
    """
    Compute the entropy for a count vector c (class counts).
    Args:
        c (array-like): class counts.
    Returns:
        float: entropy value.
    """
    s = sum(c)
    return -sum((x/s)*log2(x/s) for x in c if x > 0)

def gini(c):
    """
    Compute the Gini index for a count vector c (class counts).
    Args:
        c (array-like): class counts.
    Returns:
        float: gini index.
    """
    s = sum(c)
    return 1 - sum((x/s)**2 for x in c)

def info_gain(parent, left, right):
    """
    Compute information gain given parent, left, and right node class counts.
    Args:
        parent: class counts for the parent node
        left: class counts for the left child
        right: class counts for the right child
    Returns:
        float: information gain value
    """
    s = sum(parent)
    return entropy(parent) - (
        (sum(left)/s)*entropy(left) +
        (sum(right)/s)*entropy(right)
    )

# dataset
X, y = load_iris(return_X_y=True)
# Replace this with the dataset given if needed

# train tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

t = clf.tree_

# Print node statistics (entropy, gini, info gain)
for i in range(t.node_count):
    c = t.value[i][0]
    print(f"Node {i}:")
    print("  Counts:", c)
    print("  Entropy:", round(entropy(c), 4))
    print("  Gini:", round(gini(c), 4))

    l, r = t.children_left[i], t.children_right[i]

    if l != -1:
        left = t.value[l][0]
        right = t.value[r][0]
        print("  Info Gain:", round(info_gain(c, left, right), 4))
    print()

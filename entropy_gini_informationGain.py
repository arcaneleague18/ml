from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from math import log2

# simple functions
def entropy(c):
    s = sum(c)
    return -sum((x/s)*log2(x/s) for x in c if x>0)

def gini(c):
    s = sum(c)
    return 1 - sum((x/s)**2 for x in c)

def info_gain(parent, left, right):
    s = sum(parent)
    return entropy(parent) - (
        (sum(left)/s)*entropy(left) +
        (sum(right)/s)*entropy(right)
    )

# dataset
X, y = load_iris(return_X_y=True)
# Replace this with the dataset given

# train tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

t = clf.tree_

# print
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

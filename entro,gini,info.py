import pandas as pd
import math
#good for non-numeric data
# ---- Load the data ----
df = pd.DataFrame([
    [5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
    [4.9, 3.0, 1.4, 0.2, "Iris-setosa"],
    [5.8, 2.7, 4.1, 1.0, "Iris-versicolor"],
    [6.0, 2.2, 4.0, 1.2, "Iris-versicolor"],
    [6.9, 3.1, 4.9, 1.5, "Iris-versicolor"],
    [6.5, 3.0, 5.8, 2.2, "Iris-virginica"],
    [7.6, 3.0, 6.6, 2.1, "Iris-virginica"],
    [4.6, 3.1, 1.5, 0.2, "Iris-setosa"],
    [6.7, 3.3, 5.7, 2.5, "Iris-virginica"],
    [5.5, 2.3, 4.0, 1.3, "Iris-versicolor"],
], columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', "Species"])
# ---- Entropy function ----
def entropy(column):
    values = column.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in values)

def gini(column):
    values = column.value_counts(normalize=True)
    return 1 - sum(p**2 for p in values)

# ---- Information Gain ----
def info_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()

    weighted_entropy = 0
    for v in values:
        subset = df[df[attribute] == v]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# ---- Compute entropy ----
dataset_entropy = entropy(df["Species"])
print("Entropy of dataset:", dataset_entropy)

dataset_gini = gini(df["Species"])
print("Gini Index of dataset:", dataset_gini)

# ---- Compute information gain for each feature ----
for feature in ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]:
    ig = info_gain(df, feature, "Species")
    print(f"Information Gain for {feature}: {ig}")

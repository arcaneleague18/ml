import pandas as pd
from typing import List

def is_consistent(X, y, h: List[str]) -> bool:
    """
    Check if a hypothesis is consistent with given training data.
    Args:
        X: Feature matrix.
        y: Target labels.
        h: Hypothesis (list of attribute values or '?').
    Returns:
        bool: True if consistent, False otherwise.
    """
    for i in range(len(X)):
        match = all(h[j] == X[i][j] or h[j] == '?' for j in range(len(h)))
        if (match and y[i] == "No") or (not match and y[i] == "Yes"):
            return False  # Inconsistent if it fails even once
    return True

if __name__ == "__main__":
    # --- Step 1: Create a small dataset ---
    data = pd.DataFrame([
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # --- Step 2: Define a sample hypothesis ---
    hypothesis = ['Sunny', 'Warm', '?', 'Strong', '?', '?']

    # --- Step 4: Check consistency ---
    consistent = is_consistent(X, y, hypothesis)

    print("Hypothesis:", hypothesis)
    print("Is hypothesis consistent with training data?:", consistent)

    # --- Step 5: (Optional) Print consistency equation check for each example ---
    for i in range(len(X)):
        match = all(hypothesis[j] == X[i][j] or hypothesis[j] == '?' for j in range(len(hypothesis)))
        if (match and y[i] == "No") or (not match and y[i] == "Yes"):
            example_consistent = False
        else:
            example_consistent = True
        print(f"Example {i+1}: h(x) = {y[i]}  --> {'Consistent' if example_consistent else 'Inconsistent'}")

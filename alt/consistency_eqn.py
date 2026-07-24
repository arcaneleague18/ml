import pandas as pd
from typing import List

# --- Step 1: Create dataset ---
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

# --- Step 2: Separate features and target ---
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# --- Step 3: Define hypothesis ---
hypothesis = ['Sunny', 'Warm', '?', 'Strong', '?', '?']

def is_consistent(X, y, h: List[str]) -> bool:
    """
    Checks if hypothesis h is consistent with all training examples (X, y).
    Returns True if consistent, False otherwise.
    """
    for i in range(len(X)):
        match = all(h[j] == X[i][j] or h[j] == '?' for j in range(len(h)))
        # Inconsistent if it predicts Yes when label is No, or No when label is Yes
        if (match and y[i] == "No") or (not match and y[i] == "Yes"):
            return False
    return True

# --- Step 5: Overall consistency ---
consistent = is_consistent(X, y, hypothesis)
print("Hypothesis:", hypothesis)
print("Is hypothesis consistent with training data?:", consistent)
print("\nExample-wise consistency check:\n")

# --- Step 6: Per-example check ---
for i in range(len(X)):
    match = all(hypothesis[j] == X[i][j] or hypothesis[j] == '?' for j in range(len(hypothesis)))
    if (match and y[i] == "No") or (not match and y[i] == "Yes"):
        example_consistent = False
    else:
        example_consistent = True
    print(f"Example {i+1}: Features = {list(X[i])}, Label = {y[i]}")
    print(f"   Matches Hypothesis: {match}")
    print(f"   Example is {'Consistent' if example_consistent else 'Inconsistent'}\n")

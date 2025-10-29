import pandas as pd

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

# --- Step 4: Define consistency check function ---
def is_consistent(X, y, h):
    for i in range(len(X)):  # for each training example
        match = True
        # Check each feature one by one
        for j in range(len(h)):
            if not (h[j] == X[i][j] or h[j] == '?'):
                match = False
                break  # no need to check further for this example
        # Check for inconsistency
        if match and y[i] == "No":
            return False
        elif not match and y[i] == "Yes":
            return False
    return True

# --- Step 5: Overall consistency ---
consistent = is_consistent(X, y, hypothesis)
print("Hypothesis:", hypothesis)
print("Is hypothesis consistent with training data?:", consistent)
print("\nExample-wise consistency check:\n")

# --- Step 6: Per-example check ---
for i in range(len(X)):
    match = True
    for j in range(len(hypothesis)):
        if not (hypothesis[j] == X[i][j] or hypothesis[j] == '?'):
            match = False
            break

    # Example-level consistency
    if match and y[i] == "No":
        example_consistent = False
    elif not match and y[i] == "Yes":
        example_consistent = False
    else:
        example_consistent = True

    print(f"Example {i+1}: Features = {list(X[i])}, Label = {y[i]}")
    print(f"  → Matches Hypothesis: {match}")
    print(f"  → Example is {'Consistent' if example_consistent else 'Inconsistent'}\n")
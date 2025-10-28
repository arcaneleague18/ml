import pandas as pd

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
# Example: ['Sunny', 'Warm', '?', 'Strong', '?', '?']  (from candidate elimination result)
hypothesis = ['Sunny', 'Warm', '?', 'Strong', '?', '?']

# --- Step 3: Define consistency check function ---
def is_consistent(X, y, h):
    for i in range(len(X)):
        match = all(h[j] == X[i][j] or h[j] == '?' for j in range(len(h)))
        if (match and y[i] == "No") or (not match and y[i] == "Yes"):
            return False  # Inconsistent if it fails even once
    return True

# --- Step 4: Check consistency ---
consistent = is_consistent(X, y, hypothesis)

print("Hypothesis:", hypothesis)
print("Is hypothesis consistent with training data?:", consistent)

# --- Step 5: (Optional) Print consistency equation check for each example ---
for i in range(len(X)):
    print(f"Example {i+1}: h(x) = {y[i]}  --> {'Consistent' if consistent else 'Inconsistent'}")

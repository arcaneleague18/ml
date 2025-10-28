import pandas as pd
import numpy as np

# --- Step 1: Create dataset ---
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# --- Step 2: Candidate Elimination ---
def candidate_elimination(X, y):
    S = X[y == "Yes"][0].copy()                    # First positive example
    G = [['?' for _ in range(len(S))]]             # Most general hypothesis

    for i, val in enumerate(y):
        if val == "Yes":
            for j in range(len(S)):
                if S[j] != X[i][j]:
                    S[j] = '?'
            G = [g for g in G if all(g[k] == '?' or g[k] == S[k] for k in range(len(S)))]
        else:
            G = [g for g in G for j in range(len(S))
                 if g[j] == '?' and S[j] != X[i][j]
                 and not all(X[i][k] == g[k] or g[k] == '?' for k in range(len(S)))
                 and [S[j] if k == j else g[k] for k in range(len(S))]]
    return S, G

# --- Step 3: Run ---
S_final, G_final = candidate_elimination(X, y)
print("Most Specific Hypothesis (S):", S_final)
print("Most General Hypotheses (G):", G_final)

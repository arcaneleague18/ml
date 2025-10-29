from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#df = pd.read_csv("data.csv")
#group1 = df.iloc[:, 0]
#group2 = df.iloc[:, 1]


group1 = [12,14,15,10,13]
group2 = [8,9,12,11,7]

t_stat, p_val = stats.ttest_ind(group1, group2)
print("t-statistic:", t_stat)
print("p-value:", p_val)
plt.bar(['Group 1', 'Group 2'], [sum(group1)/len(group1), sum(group2)/len(group2)])
plt.ylabel('Mean Value')
plt.title('Mean Comparison (t-test)')
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Define two models
modelA = LogisticRegression(max_iter=200)
modelB = DecisionTreeClassifier()

# Perform cross-validation
scoresA = cross_val_score(modelA, X, y, cv=5)
scoresB = cross_val_score(modelB, X, y, cv=5)

print("Model A accuracies:", scoresA)
print("Model B accuracies:", scoresB)

# Perform paired t-test
t_stat, p_val = ttest_rel(scoresA, scoresB) #MAIN THING

print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Decision
alpha = 0.05
if p_val < alpha:
    print("✅ Reject Null Hypothesis: The models perform significantly differently.")
else:
    print("❌ Fail to Reject Null Hypothesis: No significant difference in model performance.")

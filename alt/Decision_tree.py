# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create base model
dt = DecisionTreeClassifier(random_state=42)

# Define parameter grid for optimization
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],   # splitting criteria
    'max_depth': [2, 3, 4, 5, 6, None],             # tree depth
    'min_samples_split': [2, 5, 10],                # minimum samples to split
    'min_samples_leaf': [1, 2, 4],                  # minimum samples per leaf
    'max_features': [None, 'sqrt', 'log2']          # number of features to consider
}

# Use GridSearchCV for exhaustive search
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',   # evaluation metric
    n_jobs=-1,            # use all CPU cores
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}")

# Evaluate on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")

# Visualize the best decision tree
# plt.figure(figsize=(12, 8))
# plot_tree(
#     best_model,
#     filled=True,
#     feature_names=data.feature_names,
#     class_names=data.target_names,
#     rounded=True
# )
# plt.show()

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Create dataset: two classes in 2D
class1 = np.random.randn(50, 2) + np.array([2, 2])
class2 = np.random.randn(50, 2) + np.array([6, 6])

X = np.vstack((class1, class2))
y = np.array([0]*50 + [1]*50)

# Train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nConfusion matrix\n")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy score\n")
print(accuracy_score(y_test, y_pred))

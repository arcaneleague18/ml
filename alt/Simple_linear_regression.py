from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Simple Linear Regression Example with Manual Calculation
# Dataset: x and y values
x = np.array([1, 2, 3, 4, 5, 7, 8, 9])
y = np.array([2, 4, 6, 8, 10, 14, 9, 9])

# Calculate means
xm = np.mean(x)
ym = np.mean(y)

# Calculate slope (m) and intercept (b) for best fit line y = mx + b
m = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
b = ym - xm * m

# Test data for prediction
x_test = np.array([1, 4, 6])
y_test = np.array([2, 8, 12])

def predict(x):
    """
    Predict y values given x using the calculated linear regression parameters.
    """
    return m * x + b

# Calculate Mean Squared Error (MSE) for test data
errors = y_test - predict(x_test)
mse = np.mean(errors ** 2)
print(mse)
# Alternatively:
# print(np.sum(errors**2) / len(errors))

"""
Module: alt/Simple_linear_regression.py
Purpose: Simple demo of univariate linear regression (manual and sklearn) with mean squared error calculation and basic test.
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Sequence

def predict(x: Sequence[float]) -> np.ndarray:
    """
    Predict y values given x using the calculated linear regression parameters.
    Args:
        x (array-like): Input feature(s) to predict.
    Returns:
        Predicted y values as a numpy array.
    """
    return m * np.array(x) + b

def test_simple_linear_regression() -> None:
    """
    Basic test for manual linear regression calculation.
    Verifies correct MSE for a known case and checks for edge cases.
    """
    test_x = np.array([1, 2, 3])
    test_y = np.array([2, 4, 6])
    # Compute regression parameters manually
    xm = np.mean(test_x)
    ym = np.mean(test_y)
    m_test = np.sum((test_x - xm) * (test_y - ym)) / np.sum((test_x - xm) ** 2)
    b_test = ym - xm * m_test
    def predict_test(xx):
        return m_test * xx + b_test
    preds = predict_test(test_x)
    mse_val = np.mean((test_y - preds) ** 2)
    assert np.isclose(mse_val, 0), f"MSE should be zero for perfect linear fit, got {mse_val}"
    # Edge: all y the same
    test_x2 = np.array([1,2,3])
    test_y2 = np.array([5,5,5])
    xm2 = np.mean(test_x2)
    ym2 = np.mean(test_y2)
    m2 = np.sum((test_x2 - xm2) * (test_y2 - ym2)) / np.sum((test_x2 - xm2) ** 2)
    b2 = ym2 - xm2 * m2
    assert np.isfinite(m2), "Slope should be finite"
    assert np.isfinite(b2), "Intercept should be finite"
    print("test_simple_linear_regression passed.")

# --- Main code ---
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

# Calculate Mean Squared Error (MSE) for test data
errors = y_test - predict(x_test)
mse = np.mean(errors ** 2)
print("Mean Squared Error (manual regression):", mse)
# Alternatively:
# print(np.sum(errors**2) / len(errors))

# Optional: Compare with sklearn for validation
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
sklearn_pred = model.predict(x_test.reshape(-1, 1))
sklearn_mse = mean_squared_error(y_test, sklearn_pred)
print("Mean Squared Error (sklearn regression):", sklearn_mse)

if __name__ == "__main__":
    test_simple_linear_regression()

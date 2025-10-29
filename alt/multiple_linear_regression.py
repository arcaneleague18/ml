# Multiple Linear Regression on California Housing Dataset

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print("Dataset shape:", df.shape)
print(df.head())

X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# #  Show coefficients
# coef_df = pd.DataFrame({
#     'Feature': data.feature_names,
#     'Coefficient': model.coef_
# })
# print("\nModel Coefficients:")
# print(coef_df)

#  Predict a new sample (example)
new_house = np.array([[8.3, 41, 6.5, 1.0, 950, 3.0, 35.0, -120.0]])  # example input
predicted_price = model.predict(new_house)
print(f"\nPredicted House Price (in $100,000s): {predicted_price[0]:.3f}")
